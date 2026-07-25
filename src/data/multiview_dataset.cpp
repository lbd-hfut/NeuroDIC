#include "neurodic/data/multiview_dataset.hpp"

#include "neurodic/core/exceptions.hpp"

namespace neurodic {

void MultiViewDataset::validate() const {
    if (views.empty()) {
        throw ValidationError("MultiViewDataset requires at least one view");
    }
    for (const auto& view : views) {
        view.validate();
    }
}

}  // namespace neurodic
