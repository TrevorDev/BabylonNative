#include <Babylon/Runtime.h>
#include "RuntimeImpl.h"

namespace babylon
{
    Runtime::Runtime(std::unique_ptr<RuntimeImpl> impl)
        : m_impl{ std::move(impl) }
    {
    }

    Runtime::~Runtime()
    {
    }

    void Runtime::UpdateSize(float width, float height)
    {
        m_impl->UpdateSize(width, height);
    }

    void Runtime::UpdateRenderTarget()
    {
        m_impl->UpdateRenderTarget();
    }

    void Runtime::Suspend()
    {
        m_impl->Suspend();
    }

    void Runtime::LoadScript(const std::string& url)
    {
        m_impl->LoadScript(url);
    }

    void Runtime::Eval(const std::string& string, const std::string& url)
    {
        m_impl->Eval(string, url);
    }

    void Runtime::Execute(std::function<void(Runtime&)> func)
    {
        m_impl->Execute([this, func = std::move(func)](auto&)
        {
            func(*this);
        });
    }

    babylon::Env& Runtime::Env() const
    {
        return m_impl->Env();
    }

    const std::string& Runtime::RootUrl() const
    {
        return m_impl->RootUrl();
    }
}
