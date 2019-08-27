//
// Uber, Inc. (c) 2019
//

#include "neuropods/multiprocess/control_messages.hh"

#include "gtest/gtest.h"

TEST(test_multiprocess_allowed_transitions, simple)
{
    neuropods::TransitionVerifier verifier;

    verifier.assert_transition_allowed(neuropods::LOAD_NEUROPOD);
    verifier.assert_transition_allowed(neuropods::ADD_INPUT);
    verifier.assert_transition_allowed(neuropods::ADD_INPUT);

    // A heartbeat is allowed at any time
    verifier.assert_transition_allowed(neuropods::HEARTBEAT);

    verifier.assert_transition_allowed(neuropods::ADD_INPUT);
    verifier.assert_transition_allowed(neuropods::INFER);
    verifier.assert_transition_allowed(neuropods::RETURN_OUTPUT);
    verifier.assert_transition_allowed(neuropods::RETURN_OUTPUT);

    // A heartbeat is allowed at any time
    verifier.assert_transition_allowed(neuropods::HEARTBEAT);

    verifier.assert_transition_allowed(neuropods::RETURN_OUTPUT);
    verifier.assert_transition_allowed(neuropods::END_OUTPUT);
    verifier.assert_transition_allowed(neuropods::INFER_COMPLETE);
}

TEST(test_multiprocess_allowed_transitions, shutdown)
{
    neuropods::TransitionVerifier verifier;

    verifier.assert_transition_allowed(neuropods::LOAD_NEUROPOD);
    verifier.assert_transition_allowed(neuropods::ADD_INPUT);
    verifier.assert_transition_allowed(neuropods::ADD_INPUT);

    // Shutdown is allowed at any time
    verifier.assert_transition_allowed(neuropods::SHUTDOWN);
}

TEST(test_multiprocess_allowed_transitions, invalid_start)
{
    neuropods::TransitionVerifier verifier;

    // Infer is not allowed here
    EXPECT_ANY_THROW(verifier.assert_transition_allowed(neuropods::INFER));
}

TEST(test_multiprocess_allowed_transitions, invalid)
{
    neuropods::TransitionVerifier verifier;

    verifier.assert_transition_allowed(neuropods::LOAD_NEUROPOD);
    verifier.assert_transition_allowed(neuropods::ADD_INPUT);
    verifier.assert_transition_allowed(neuropods::ADD_INPUT);

    // This is invalid
    EXPECT_ANY_THROW(verifier.assert_transition_allowed(neuropods::INFER_COMPLETE));
}

TEST(test_multiprocess_allowed_transitions, new_infer)
{
    neuropods::TransitionVerifier verifier;

    // Load a neuropod and run inference
    verifier.assert_transition_allowed(neuropods::LOAD_NEUROPOD);
    verifier.assert_transition_allowed(neuropods::ADD_INPUT);
    verifier.assert_transition_allowed(neuropods::INFER);
    verifier.assert_transition_allowed(neuropods::RETURN_OUTPUT);
    verifier.assert_transition_allowed(neuropods::END_OUTPUT);
    verifier.assert_transition_allowed(neuropods::INFER_COMPLETE);

    // Running inference again is valid
    verifier.assert_transition_allowed(neuropods::ADD_INPUT);
}

TEST(test_multiprocess_allowed_transitions, load_neuropod)
{
    neuropods::TransitionVerifier verifier;

    // Load a neuropod and run inference
    verifier.assert_transition_allowed(neuropods::LOAD_NEUROPOD);
    verifier.assert_transition_allowed(neuropods::ADD_INPUT);
    verifier.assert_transition_allowed(neuropods::INFER);
    verifier.assert_transition_allowed(neuropods::RETURN_OUTPUT);
    verifier.assert_transition_allowed(neuropods::END_OUTPUT);
    verifier.assert_transition_allowed(neuropods::INFER_COMPLETE);

    // Loading another neuropod is valid
    verifier.assert_transition_allowed(neuropods::LOAD_NEUROPOD);
}
