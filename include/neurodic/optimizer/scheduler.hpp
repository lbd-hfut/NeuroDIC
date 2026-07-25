/**
 * Learning-rate scheduler shell.
 *
 * Responsibilities: adjust optimizer hyperparameters during training.
 * Inputs: iteration metrics.
 * Outputs: updated learning-rate state.
 * Ownership: value shell.
 * Differentiable: NO.
 * TODO(NeuroDIC): define scheduler policies after baseline optimizer works.
 */
#pragma once

namespace neurodic { class Scheduler { public: void step() {} }; }
