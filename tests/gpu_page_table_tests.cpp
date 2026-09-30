#include "test_support.h"
#include "page_table.h"
#include "minillm/cuda/kv_layout.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <new>
#include <optional>
#include <random>
#include <set>

using namespace minillm::cuda;

namespace {
bool reject_allocation = false;
struct NoAllocation {
    NoAllocation() { reject_allocation = true; }
    ~NoAllocation() { reject_allocation = false; }
};
template<class T> std::vector<T> copy(std::span<const T> values) {
    return {values.begin(), values.end()};
}
std::size_t pages(std::size_t length) { return (length + 15) / 16; }
std::set<PhysicalPageId> mapped(std::span<const PhysicalPageId> table) {
    std::set<PhysicalPageId> result;
    for (const auto id : table) {
        if (id != invalid_page) { CHECK(result.insert(id).second); }
    }
    return result;
}
void partition(std::span<const PhysicalPageId> table, const PageTableState& state,
               const PageTableLimits& limits, std::span<const std::size_t> lengths) {
    std::set<PhysicalPageId> used;
    CHECK(table.size() == lengths.size() * state.blocks_per_sequence());
    for (std::size_t s = 0; s < lengths.size(); ++s) {
        for (std::size_t b = 0; b < state.blocks_per_sequence(); ++b) {
            const auto id = table[s * state.blocks_per_sequence() + b];
            if (b < pages(lengths[s])) {
                CHECK(id >= 0 && std::size_t(id) < limits.physical_pages);
                CHECK(used.insert(id).second);
            } else { CHECK(id == invalid_page); }
        }
    }
    CHECK(used.size() == state.assigned_pages());
    CHECK(used.size() + state.free_pages() == limits.physical_pages);
    for (const auto id : state.free_page_ids()) {
        CHECK(id >= 0 && std::size_t(id) < limits.physical_pages);
        CHECK(used.insert(id).second);
    }
    CHECK(used.size() == limits.physical_pages);
}
}

// 页事务只使用普通对齐的整数 vector；禁用 new 可验证预留空间真正被复用。
void* operator new(std::size_t bytes) {
    if (reject_allocation) { throw std::bad_alloc(); }
    if (auto* pointer = std::malloc(bytes ? bytes : 1)) { return pointer; }
    throw std::bad_alloc();
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }

TEST(kv_capacity_contract_and_overflow_preflight) {
    CHECK(kv_layout_name(CudaKvLayout::contiguous) == "contiguous");
    CHECK(kv_layout_name(CudaKvLayout::paged) == "paged");
    CHECK(checked_kv_capacity(CudaKvLayout::contiguous,4,2048,0,16) == 8192);
    CHECK(checked_kv_capacity(CudaKvLayout::contiguous,4,2048,8192,16) == 8192);
    CHECK(checked_kv_capacity(CudaKvLayout::paged,4,2048,2560,16) == 2560);
    CHECK(checked_kv_capacity(CudaKvLayout::paged,1,17,32,16) == 32);
    for (auto capacity : {std::size_t{0},std::size_t{17},std::size_t{8208},
                          std::numeric_limits<std::size_t>::max()}) {
        test::throws<std::invalid_argument>([&] {
            checked_kv_capacity(CudaKvLayout::paged,4,2048,capacity,16);
        });
    }
    test::throws<std::invalid_argument>([] { checked_kv_capacity(CudaKvLayout::contiguous,4,2048,2560,16); });
    test::throws<std::invalid_argument>([] { checked_kv_capacity(static_cast<CudaKvLayout>(99),4,2048,0,16); });
    for (const auto limits : {PageTableLimits{0,16,16,1}, {5,16,16,1}, {1,0,16,1},
                              {1,2049,16,1}, {1,16,0,1}, {1,16,32,1}, {1,16,16,0},
                              {1,16,16,2}, {4,2048,16,513},
                              {SIZE_MAX,SIZE_MAX,SIZE_MAX,SIZE_MAX}}) {
        test::throws<std::invalid_argument>([&] { PageTableState state(limits); });
    }
    PageTableState maximum({4,2048,16,512});
    CHECK(maximum.committed_table().size() * sizeof(PhysicalPageId) == 2048);
}

TEST(page_boundaries_prepare_reserve_commit_and_clear) {
    const PageTableLimits limits{1,65,16,5};
    PageTableState state(limits);
    std::array<std::size_t,1> current{0};
    CHECK(state.upload_required());
    for (const auto length : {1u,15u,16u,17u,31u,32u,33u,64u,65u}) {
        const std::array<std::size_t,1> next{length};
        const auto before = copy(state.committed_table());
        const auto free_before = copy(state.free_page_ids());
        const auto needed = pages(length)-pages(current[0]);
        state.prepare(current,next);
        CHECK(state.phase() == PageTablePhase::prepared && state.planned_pages() == needed);
        CHECK(copy(state.committed_table()) == before && copy(state.free_page_ids()) == free_before);
        CHECK(state.upload_required() == (current[0] == 0 || needed != 0));
        state.begin_execution();
        CHECK(state.phase() == PageTablePhase::executing);
        partition(state.device_table_for_upload(),state,limits,next);
        state.commit();
        CHECK(state.phase() == PageTablePhase::ready && !state.upload_required());
        current = next;
        partition(state.committed_table(),state,limits,current);
    }
    state.clear(0);
    current[0] = 0;
    CHECK(state.upload_required());
    partition(state.committed_table(),state,limits,current);
    state.clear(0);
    CHECK(state.free_pages() == 5);
}

TEST(page_capacity_failure_and_invalid_lengths_are_atomic) {
    PageTableState state({4,33,16,4});
    const std::array<std::size_t,4> empty{}, next{1,1,1,1}, exhausted{17,17,17,17};
    state.prepare(empty,next); state.begin_execution(); state.commit();
    const auto before = copy(state.committed_table()), free_before = copy(state.free_page_ids());
    test::throws<std::length_error>([&] { state.prepare(next,exhausted); });
    for (const auto invalid : {std::array<std::size_t,4>{0,1,1,1}, {1,1,1,34},
                               {1,1,SIZE_MAX,1}}) {
        test::throws<std::invalid_argument>([&] { state.prepare(next,invalid); });
    }
    test::throws<std::invalid_argument>([&] { state.prepare(std::array<std::size_t,3>{},next); });
    test::throws<std::invalid_argument>([&] { state.prepare(empty,next); });
    CHECK(state.phase() == PageTablePhase::ready && !state.upload_required());
    CHECK(copy(state.committed_table()) == before && copy(state.free_page_ids()) == free_before);
    test::throws<std::invalid_argument>([&] { state.clear(4); });
    CHECK(copy(state.committed_table()) == before && state.free_pages() == 0);
}

TEST(discard_preserves_free_order_and_dirty_clear) {
    PageTableState state({2,32,16,4});
    const std::array<std::size_t,2> empty{}, one{16,0}, two{17,1};
    state.prepare(empty,one); state.begin_execution(); state.commit();
    const auto active = copy(state.committed_table()), free = copy(state.free_page_ids());
    state.prepare(one,two);
    const auto tentative = copy(state.device_table_for_upload());
    state.discard_prepared();
    CHECK(!state.upload_required() && copy(state.committed_table()) == active);
    CHECK(copy(state.free_page_ids()) == free);
    state.prepare(one,two);
    CHECK(copy(state.device_table_for_upload()) == tentative);
    state.discard_prepared();
    state.clear(0);
    state.prepare(empty,empty);
    CHECK(state.upload_required());
    state.discard_prepared();
    CHECK(state.upload_required());
    state.prepare(empty,empty); state.begin_execution(); state.commit();
    CHECK(!state.upload_required() && state.assigned_pages() == 0);
}

TEST(clear_reuses_exclusive_pages_across_sequences) {
    const PageTableLimits limits{4,32,16,4};
    PageTableState state(limits);
    std::array<std::size_t,4> lengths{}, next{16,0,17,0};
    state.prepare(lengths,next); state.begin_execution(); state.commit();
    const auto before = copy(state.committed_table());
    state.clear(2);
    lengths = {16,0,0,0};
    next = {16,17,0,0};
    state.prepare(lengths,next); state.begin_execution(); state.commit();
    const auto after = copy(state.committed_table());
    CHECK(after[0] == before[0]);
    CHECK((std::set<PhysicalPageId>{after[2],after[3]} ==
           std::set<PhysicalPageId>{before[4],before[5]}));
    partition(state.committed_table(),state,limits,next);
}

TEST(postlaunch_poison_quarantines_old_and_new_pages) {
    const PageTableLimits limits{2,32,16,4};
    PageTableState state(limits);
    const std::array<std::size_t,2> empty{}, one{1,0}, next{17,1};
    state.prepare(empty,one); state.begin_execution(); state.commit();
    const auto committed = copy(state.committed_table());
    state.prepare(one,next); state.begin_execution();
    const auto free = copy(state.free_page_ids());
    state.poison();
    CHECK(state.phase() == PageTablePhase::poisoned && state.assigned_pages() == 3);
    CHECK(copy(state.committed_table()) == committed && copy(state.free_page_ids()) == free);
    partition(state.device_table_for_upload(),state,limits,next);
    test::throws<std::logic_error>([&] { state.clear(0); });
    test::throws<std::logic_error>([&] { state.prepare(one,next); });
    state.discard_prepared(); state.commit(); state.begin_execution(); state.poison();
    CHECK(state.phase() == PageTablePhase::poisoned && copy(state.free_page_ids()) == free);
}

TEST(illegal_phase_transitions_fail_closed) {
    PageTableState unprepared({1,16,16,1});
    unprepared.begin_execution();
    CHECK(unprepared.phase() == PageTablePhase::poisoned);
    PageTableState state({1,16,16,1});
    state.prepare(std::array<std::size_t,1>{0},std::array<std::size_t,1>{1});
    test::throws<std::logic_error>([&] { state.clear(0); });
    test::throws<std::logic_error>([&] {
        state.prepare(std::array<std::size_t,1>{0},std::array<std::size_t,1>{1});
    });
    state.commit();
    CHECK(state.phase() == PageTablePhase::poisoned && state.free_pages() == 0);
    CHECK(state.committed_table()[0] == invalid_page);
}

TEST(legal_transactions_allocate_only_at_construction) {
    PageTableState state({4,65,16,20});
    const std::array<std::size_t,4> empty{}, full{65,65,65,65};
    {
        NoAllocation guard;
        state.prepare(empty,full); state.discard_prepared();
        state.prepare(empty,full); state.begin_execution(); state.commit();
        for (std::size_t s = 0; s < 4; ++s) { state.clear(s); state.clear(s); }
        state.prepare(empty,full); state.begin_execution(); state.poison();
    }
    CHECK(state.phase() == PageTablePhase::poisoned && state.free_pages() == 0);
}

TEST(fixed_seed_ten_thousand_operations_match_set_oracle) {
    const PageTableLimits limits{4,65,16,13};
    std::optional<PageTableState> state(std::in_place,limits);
    std::array<std::size_t,4> lengths{};
    std::mt19937 rng(20260928);
    std::size_t failures = 0, discarded = 0, poisoned = 0, committed = 0;
    for (std::size_t step = 0; step < 10000; ++step) {
        const auto operation = rng() % 8;
        if (operation == 0) {
            const auto s = rng() % 4;
            state->clear(s); lengths[s] = 0;
        } else {
            auto next = lengths;
            for (auto& length : next) {
                if (rng() % 2) { length = std::min(limits.max_model_len, length + rng() % 19); }
            }
            std::size_t needed = 0;
            for (std::size_t s = 0; s < 4; ++s) { needed += pages(next[s])-pages(lengths[s]); }
            const auto before = copy(state->committed_table()), free = copy(state->free_page_ids());
            if (needed > free.size()) {
                test::throws<std::length_error>([&] { state->prepare(lengths,next); });
                ++failures;
            } else {
                state->prepare(lengths,next);
                const auto table = state->device_table_for_upload();
                const auto assigned = mapped(table);
                CHECK(assigned.size() == limits.physical_pages-free.size()+needed);
                CHECK(state->planned_pages() == needed && copy(state->free_page_ids()) == free);
                const std::set<PhysicalPageId> free_set(free.begin(),free.end());
                for (std::size_t i = 0; i < table.size(); ++i) {
                    if (before[i] != invalid_page) { CHECK(table[i] == before[i]); }
                    else if (table[i] != invalid_page) { CHECK(free_set.contains(table[i])); }
                }
                if (operation == 1) {
                    state->discard_prepared();
                    ++discarded;
                } else {
                    state->begin_execution();
                    partition(table,*state,limits,next);
                    if (operation == 2) {
                        state->poison();
                        partition(table,*state,limits,next);
                        test::throws<std::logic_error>([&] { state->clear(0); });
                        state.emplace(limits); lengths = {};
                        ++poisoned;
                    } else {
                        state->commit(); lengths = next;
                        ++committed;
                    }
                }
            }
            if (needed > free.size() || operation == 1) {
                CHECK(copy(state->committed_table()) == before && copy(state->free_page_ids()) == free);
            }
        }
        CHECK(state->phase() == PageTablePhase::ready);
        partition(state->committed_table(),*state,limits,lengths);
    }
    CHECK(failures > 100 && discarded > 100 && poisoned > 100 && committed > 100);
}

int main() { return test::run(); }
