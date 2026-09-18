#include "llmserve/utf8.h"

#include <cstddef>

namespace llmserve {

std::string Utf8Buffer::append(std::string_view bytes) {
    pending_.append(bytes);
    return drain(false);
}

std::string Utf8Buffer::finish() {
    return drain(true);
}

std::string Utf8Buffer::drain(bool final) {
    std::string output;
    std::size_t pos = 0;
    while (pos < pending_.size()) {
        const auto lead = static_cast<unsigned char>(pending_[pos]);
        std::size_t width = 0;
        if (lead < 0x80) {
            width = 1;
        } else if (lead >= 0xc2 && lead <= 0xdf) {
            width = 2;
        } else if (lead >= 0xe0 && lead <= 0xef) {
            width = 3;
        } else if (lead >= 0xf0 && lead <= 0xf4) {
            width = 4;
        }
        bool valid = width != 0;
        const auto available = pending_.size() - pos;
        for (std::size_t i = 1; valid && i < width && i < available; ++i) {
            const auto byte = static_cast<unsigned char>(pending_[pos + i]);
            valid = byte >= 0x80 && byte <= 0xbf;
            if (i == 1) {
                valid = valid && !(lead == 0xe0 && byte < 0xa0) &&
                    !(lead == 0xed && byte >= 0xa0) && !(lead == 0xf0 && byte < 0x90) &&
                    !(lead == 0xf4 && byte >= 0x90);
            }
        }
        if (valid && available < width && !final) {
            break;
        }
        if (!valid || available < width) {
            output += "\xef\xbf\xbd";
            pos += valid ? available : 1;
        } else {
            output.append(pending_, pos, width);
            pos += width;
        }
    }
    pending_.erase(0, pos);
    return output;
}

} // namespace llmserve
