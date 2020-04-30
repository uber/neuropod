//
// Uber, Inc. (c) 2020
//

#include "neuropod/multiprocess/shm/pool_shm_allocator.hh"

#include "neuropod/internal/error_utils.hh"
#include "neuropod/internal/memory_utils.hh"

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/smart_ptr/shared_ptr.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <iostream>
#include <mutex>
#include <unordered_map>

namespace neuropod
{

namespace
{

namespace ipc = boost::interprocess;

template <class T, class SegmentManager>
class deleter
{
public:
    typedef typename boost::intrusive::pointer_traits<typename SegmentManager::void_pointer>::template rebind_pointer<
        T>::type pointer;

private:
    typedef typename boost::intrusive::pointer_traits<pointer>::template rebind_pointer<SegmentManager>::type
        segment_manager_pointer;

    segment_manager_pointer mp_mngr;

public:
    deleter(segment_manager_pointer pmngr) : mp_mngr(pmngr) {}

    void operator()(const pointer &p) { mp_mngr->deallocate(ipc::ipcdetail::to_raw_pointer(p)); }
};

typedef ipc::managed_shared_memory::segment_manager                 segment_manager_type;
typedef ipc::allocator<void, segment_manager_type>                  void_allocator_type;
typedef deleter<void, segment_manager_type>                         deleter_type;
typedef ipc::shared_ptr<void, void_allocator_type, deleter_type>    my_shared_ptr;

thread_local boost::uuids::random_generator uuid_generator;

} // namespace

RawSHMBlockAllocator::RawSHMBlockAllocator() : pool_(std::make_shared<Pool>()) {}

RawSHMBlockAllocator::~RawSHMBlockAllocator() = default;

struct RawSHMBlockAllocator::Pool
{
    // 100 MB pool
    ipc::managed_shared_memory segment;

    Pool() : segment(ipc::open_or_create, "MySharedMemory", 1024 * 1024 * 100) {}
};

std::shared_ptr<void> RawSHMBlockAllocator::allocate_shm(size_t size_bytes, RawSHMHandle &handle)
{
    // Generate a UUID
    auto uuid     = uuid_generator();
    auto uuid_str = boost::uuids::to_string(uuid);

    // Set the handle
    memcpy(handle.data(), uuid.data, sizeof(uuid.data));

    // Allocate memory
    void * block = pool_->segment.allocate(size_bytes);

    // Create an ipc shared pointer in shared memory
    auto shared_instance = pool_->segment.construct<my_shared_ptr>(uuid_str.c_str())(
        block,
        void_allocator_type(pool_->segment.get_segment_manager()),
        deleter_type(pool_->segment.get_segment_manager()));

    // Sanity check
    assert(shared_instance->use_count() == 1);

    void *data = shared_instance->get().get();

    auto &pool = pool_;
    return std::shared_ptr<void>(data, [pool, shared_instance](void *unused) {
        // Destroy the ipc shared ptr
        pool->segment.destroy_ptr(shared_instance);
    });
}

std::shared_ptr<void> RawSHMBlockAllocator::load_shm(const RawSHMHandle &handle)
{
    // Get the id
    boost::uuids::uuid uuid;
    memcpy(uuid.data, handle.data(), sizeof(uuid.data));

    // Convert to string
    auto uuid_str = boost::uuids::to_string(uuid);

    // Get the shared pointer
    auto shared_instance = pool_->segment.find<my_shared_ptr>(uuid_str.c_str()).first;

    // Sanity check
    if (!shared_instance)
    {
        // This means that the other process isn't keeping references to data long enough for this
        // process to load the data.
        // This can lead to some hard to debug race conditions so we always throw an error.
        NEUROPOD_ERROR("Tried getting a pointer to an existing chunk of memory that has a refcount of zero: {}", uuid_str);
    }

    // Create another shared pointer in shared memory
    // Generate another UUID
    auto uuid2           = uuid_generator();
    auto uuid2_str       = boost::uuids::to_string(uuid2);
    auto copied_instance = pool_->segment.construct<my_shared_ptr>(uuid2_str.c_str())(*shared_instance);

    void *data = shared_instance->get().get();

    // Create a shared pointer to the underlying data with a custom deleter
    // that keeps the block alive
    auto &pool = pool_;
    return std::shared_ptr<void>(data, [pool, copied_instance](void *unused) {
        // Destroy the ipc shared ptr
        pool->segment.destroy_ptr(copied_instance);
    });
}

} // namespace neuropod
