#include "neurodic/data/image.hpp"

#include "neurodic/core/exceptions.hpp"

namespace neurodic {

Image::Image(torch::Tensor tensor) : tensor_(std::move(tensor)) {}

std::int64_t Image::width() const {
    validate();
    return tensor_.size(-1);
}

std::int64_t Image::height() const {
    validate();
    return tensor_.size(-2);
}

std::int64_t Image::channels() const {
    validate();
    return tensor_.dim() == 2 ? 1 : tensor_.size(-3);
}

void Image::validate() const {
    if (!tensor_.defined()) {
        throw ValidationError("Image tensor must be defined");
    }
    if (tensor_.dim() != 2 && tensor_.dim() != 3) {
        throw ValidationError("Image tensor must have shape [H, W] or [C, H, W]");
    }
}

}  // namespace neurodic
