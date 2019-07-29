//
// Uber, Inc. (c) 2019
//

#include "neuropods/internal/neuropod_tensor.hh"

namespace neuropods
{

NeuropodTensor *NeuropodValue::as_tensor()
{
    assure_tensor();
    return dynamic_cast<NeuropodTensor *>(this);
}

const NeuropodTensor *NeuropodValue::as_tensor() const
{
    assure_tensor();
    return dynamic_cast<const NeuropodTensor *>(this);
}

template <typename T>
TypedNeuropodTensor<T> *NeuropodValue::as_typed_tensor()
{
    return this->as_tensor()->as_typed_tensor<T>();
}

template <typename T>
const TypedNeuropodTensor<T> *NeuropodValue::as_typed_tensor() const
{
    return this->as_tensor()->as_typed_tensor<T>();
}

#define INIT_TEMPLATES_FOR_TYPE(CPP_TYPE, NEUROPOD_TYPE)                            \
    template TypedNeuropodTensor<CPP_TYPE> *      NeuropodValue::as_typed_tensor(); \
    template const TypedNeuropodTensor<CPP_TYPE> *NeuropodValue::as_typed_tensor() const;

FOR_EACH_TYPE_MAPPING_INCLUDING_STRING(INIT_TEMPLATES_FOR_TYPE);

} // namespace neuropods
