#include "neurodic/core/context.hpp"

#include "neurodic/core/exceptions.hpp"

namespace neurodic {

void RuntimeContext::validate() const {
    if (device.empty()) {
        throw ValidationError("RuntimeContext.device must not be empty");
    }
}

}  // namespace neurodic
