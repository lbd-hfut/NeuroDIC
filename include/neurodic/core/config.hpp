/**
 * Configuration shell.
 *
 * Responsibilities: represent parsed user/runtime configuration without
 * committing to a final schema.
 * Inputs: future YAML/Python dictionaries.
 * Outputs: typed problem-builder options.
 * Ownership: value object.
 * Differentiable: NO.
 * TODO(NeuroDIC): replace placeholder fields with a validated schema.
 */
#pragma once

#include <string>

namespace neurodic {

struct Config {
    std::string name;
    void validate() const {}
};

}  // namespace neurodic
