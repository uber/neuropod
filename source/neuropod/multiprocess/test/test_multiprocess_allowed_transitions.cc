/* Copyright (c) 2020 The Neuropod Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "gtest/gtest.h"
#include "neuropod/multiprocess/ipc_control_channel.hh"

TEST(test_multiprocess_allowed_transitions, simple)
{
    neuropod::TransitionVerifier verifier;

    verifier.assert_transition_allowed(neuropod::LOAD_NEUROPOD);
    verifier.assert_transition_allowed(neuropod::LOAD_SUCCESS);
    verifier.assert_transition_allowed(neuropod::ADD_INPUT);
    verifier.assert_transition_allowed(neuropod::INFER);
    verifier.assert_transition_allowed(neuropod::RETURN_OUTPUT);
}

TEST(test_multiprocess_allowed_transitions, shutdown)
{
    neuropod::TransitionVerifier verifier;

    verifier.assert_transition_allowed(neuropod::LOAD_NEUROPOD);
    verifier.assert_transition_allowed(neuropod::LOAD_SUCCESS);
    verifier.assert_transition_allowed(neuropod::ADD_INPUT);

    // Shutdown is allowed at any time
    verifier.assert_transition_allowed(neuropod::SHUTDOWN);
}

TEST(test_multiprocess_allowed_transitions, invalid_start)
{
    neuropod::TransitionVerifier verifier;

    // Infer is not allowed here
    EXPECT_ANY_THROW(verifier.assert_transition_allowed(neuropod::INFER));
}

TEST(test_multiprocess_allowed_transitions, invalid)
{
    neuropod::TransitionVerifier verifier;

    verifier.assert_transition_allowed(neuropod::LOAD_NEUROPOD);
    verifier.assert_transition_allowed(neuropod::LOAD_SUCCESS);
    verifier.assert_transition_allowed(neuropod::ADD_INPUT);

    // This is invalid
    EXPECT_ANY_THROW(verifier.assert_transition_allowed(neuropod::LOAD_NEUROPOD));
}

TEST(test_multiprocess_allowed_transitions, new_infer)
{
    neuropod::TransitionVerifier verifier;

    // Load a neuropod and run inference
    verifier.assert_transition_allowed(neuropod::LOAD_NEUROPOD);
    verifier.assert_transition_allowed(neuropod::LOAD_SUCCESS);
    verifier.assert_transition_allowed(neuropod::ADD_INPUT);
    verifier.assert_transition_allowed(neuropod::INFER);
    verifier.assert_transition_allowed(neuropod::RETURN_OUTPUT);

    // Running inference again is valid
    verifier.assert_transition_allowed(neuropod::ADD_INPUT);
}

TEST(test_multiprocess_allowed_transitions, load_neuropod)
{
    neuropod::TransitionVerifier verifier;

    // Load a neuropod and run inference
    verifier.assert_transition_allowed(neuropod::LOAD_NEUROPOD);
    verifier.assert_transition_allowed(neuropod::LOAD_SUCCESS);
    verifier.assert_transition_allowed(neuropod::ADD_INPUT);
    verifier.assert_transition_allowed(neuropod::INFER);
    verifier.assert_transition_allowed(neuropod::RETURN_OUTPUT);

    // Loading another neuropod is valid
    verifier.assert_transition_allowed(neuropod::LOAD_NEUROPOD);
}
