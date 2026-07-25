/**
 * NeuroDIC exception hierarchy.
 *
 * Responsibilities: provide domain-specific exception types for architecture
 * and validation failures.
 * Inputs: error messages.
 * Outputs: typed exceptions.
 * Ownership: exceptions own their message strings.
 * Differentiable: NO.
 * TODO(NeuroDIC): refine error categories once algorithms are implemented.
 */
#pragma once

#include <stdexcept>
#include <string>

namespace neurodic {

class NeuroDICError : public std::runtime_error {
public:
    explicit NeuroDICError(const std::string& message) : std::runtime_error(message) {}
};

class ValidationError : public NeuroDICError {
public:
    explicit ValidationError(const std::string& message) : NeuroDICError(message) {}
};

class NotImplementedScientificError : public NeuroDICError {
public:
    explicit NotImplementedScientificError(const std::string& message) : NeuroDICError(message) {}
};

}  // namespace neurodic
