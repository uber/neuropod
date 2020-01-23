//
// Uber, Inc. (c) 2020
//

#pragma once

#include "neuropod/internal/double_buffered_queue.hh"

#include <atomic>
#include <thread>
#include <type_traits>

namespace neuropod
{

// A worker thread that runs a lambda in a loop
class WorkerThreadWithLoop
{
private:
    std::atomic_bool enable_;
    std::thread      worker_thread_;

    // Runs `lambda` and returns its return value
    template <typename Lambda>
    static typename std::enable_if<std::is_same<typename std::result_of<Lambda()>::type, bool>::value, bool>::type call(
        Lambda lambda)
    {
        return lambda();
    }

    // Runs `lambda` and returns true
    template <typename Lambda>
    static typename std::enable_if<!std::is_same<typename std::result_of<Lambda()>::type, bool>::value, bool>::type
    call(Lambda lambda)
    {
        lambda();

        return true;
    }

public:
    template <typename Lambda>
    WorkerThreadWithLoop(Lambda loop_body)
        : enable_(true), worker_thread_([this, loop_body] {
              while (enable_)
              {
                  // Run the loop body and break if we need to
                  if (!call(loop_body))
                  {
                      break;
                  }
              }
          })
    {
    }

    virtual ~WorkerThreadWithLoop()
    {
        // Disable the worker thread and join
        enable_ = false;
        worker_thread_.join();
    }
};

// A worker thread that processes items from an input queue
template <typename InputType, template <typename QueueInputType> class InputQueue = DoubleBufferedQueue>
class WorkerThreadWithInputQueue
{
private:
    WorkerThreadWithLoop  worker_thread_;
    InputQueue<InputType> input_queue_;

public:
    template <typename Lambda>
    WorkerThreadWithInputQueue(Lambda handler)
        : worker_thread_([this, handler] {
              // Get an item from the queue
              InputType item;
              bool      success = input_queue_.dequeue(item);

              // Since we're waiting, the only time success is false is when we're shutting down
              if (!success)
              {
                  return false;
              }

              // Process the item
              return handler(item);
          })
    {
    }

    template <typename... Params>
    void enqueue(Params &&... params)
    {
        input_queue_.enqueue(std::forward<Params>(params)...);
    }

    ~WorkerThreadWithInputQueue() { input_queue_.shutdown(); }
};

// A worker thread that processes items from an input queue and returns items to an output queue
template <typename InputType,
          typename OutputType,
          template <typename QueueInputType> class InputQueue  = DoubleBufferedQueue,
          template <typename QueueInputType> class OutputQueue = DoubleBufferedQueue>
class WorkerThreadWithInputAndOutputQueue
{
private:
    WorkerThreadWithInputQueue<InputType, InputQueue> worker_thread_;
    OutputQueue<OutputType>                           output_queue_;

public:
    template <typename Lambda>
    WorkerThreadWithInputAndOutputQueue(Lambda handler)
        : worker_thread_([this, handler](InputType &item) {
              // Process the item
              OutputType output;
              bool       did_return = handler(item, output);

              // Add to the output queue if we need to
              if (did_return)
              {
                  output_queue_.enqueue(std::move(output));
              }

              return true;
          })
    {
    }

    template <typename... Params>
    void enqueue(Params &&... params)
    {
        worker_thread_.enqueue(std::forward<Params>(params)...);
    }

    template <typename... Params>
    bool dequeue(Params &&... params)
    {
        return output_queue_.dequeue(std::forward<Params>(params)...);
    }

    ~WorkerThreadWithInputAndOutputQueue() { output_queue_.shutdown(); }
};

} // namespace neuropod