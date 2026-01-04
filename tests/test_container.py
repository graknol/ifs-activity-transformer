"""
Unit tests for service container (services/container.py).

Tests the dependency injection container implementation.
"""
import pytest
from services.container import ServiceContainer, ServiceProvider


class TestServiceContainer:
    """Test ServiceContainer class."""
    
    def test_register_and_resolve_singleton(self, service_container):
        """Test registering and resolving singleton services."""
        # Create a test service
        test_service = {"name": "test", "value": 42}
        service_container.register_singleton('test_service', test_service)
        
        # Resolve should return the same instance
        resolved = service_container.resolve('test_service')
        assert resolved is test_service
        assert resolved['value'] == 42
    
    def test_singleton_returns_same_instance(self, service_container):
        """Test that singleton returns the same instance every time."""
        test_service = {"counter": 0}
        service_container.register_singleton('counter', test_service)
        
        # Multiple resolves should return the same instance
        instance1 = service_container.resolve('counter')
        instance2 = service_container.resolve('counter')
        
        assert instance1 is instance2
        
        # Modifying one affects the other
        instance1['counter'] = 5
        assert instance2['counter'] == 5
    
    def test_register_and_resolve_transient(self, service_container):
        """Test registering and resolving transient services."""
        # Register a factory that creates new instances
        call_count = [0]
        
        def factory():
            call_count[0] += 1
            return {"instance": call_count[0]}
        
        service_container.register_transient('transient_service', factory)
        
        # Each resolve should call the factory
        instance1 = service_container.resolve('transient_service')
        instance2 = service_container.resolve('transient_service')
        
        assert instance1['instance'] == 1
        assert instance2['instance'] == 2
        assert instance1 is not instance2
    
    def test_register_scoped(self, service_container):
        """Test registering scoped services."""
        call_count = [0]
        
        def factory():
            call_count[0] += 1
            return {"instance": call_count[0]}
        
        service_container.register_scoped('scoped_service', factory)
        
        # Should behave like transient without a ServiceProvider
        instance1 = service_container.resolve('scoped_service')
        instance2 = service_container.resolve('scoped_service')
        
        assert instance1['instance'] == 1
        assert instance2['instance'] == 2
    
    def test_resolve_nonexistent_service(self, service_container):
        """Test resolving a service that doesn't exist."""
        with pytest.raises(ValueError, match="Service 'nonexistent' is not registered"):
            service_container.resolve('nonexistent')
    
    def test_is_registered(self, service_container):
        """Test checking if a service is registered."""
        assert not service_container.is_registered('test_service')
        
        service_container.register_singleton('test_service', "test")
        assert service_container.is_registered('test_service')
        
        service_container.register_transient('transient_service', lambda: "test")
        assert service_container.is_registered('transient_service')
    
    def test_clear(self, service_container):
        """Test clearing all registered services."""
        service_container.register_singleton('singleton', "test1")
        service_container.register_transient('transient', lambda: "test2")
        
        assert service_container.is_registered('singleton')
        assert service_container.is_registered('transient')
        
        service_container.clear()
        
        assert not service_container.is_registered('singleton')
        assert not service_container.is_registered('transient')
    
    def test_register_lambda_factory(self, service_container):
        """Test registering services with lambda factories."""
        service_container.register_transient('lambda_service', lambda: {"value": 100})
        
        resolved = service_container.resolve('lambda_service')
        assert resolved['value'] == 100
    
    def test_register_class_factory(self, service_container):
        """Test registering services with class factories."""
        class TestService:
            def __init__(self):
                self.name = "TestService"
        
        service_container.register_transient('class_service', TestService)
        
        resolved = service_container.resolve('class_service')
        assert isinstance(resolved, TestService)
        assert resolved.name == "TestService"


class TestServiceProvider:
    """Test ServiceProvider class for scoped services."""
    
    def test_service_provider_context_manager(self, service_container):
        """Test ServiceProvider as context manager."""
        service_container.register_singleton('singleton', {"value": 1})
        
        with ServiceProvider(service_container) as provider:
            resolved = provider.get_service('singleton')
            assert resolved['value'] == 1
    
    def test_scoped_service_caching(self, service_container):
        """Test that scoped services are cached within a scope."""
        call_count = [0]
        
        def factory():
            call_count[0] += 1
            return {"instance": call_count[0]}
        
        service_container.register_transient('scoped_service', factory)
        
        with ServiceProvider(service_container) as provider:
            provider.register_scoped_service('scoped_service')
            
            # Within the same scope, should return the same instance
            instance1 = provider.get_service('scoped_service')
            instance2 = provider.get_service('scoped_service')
            
            assert instance1 is instance2
            assert instance1['instance'] == 1
        
        # New scope should create a new instance
        with ServiceProvider(service_container) as provider:
            provider.register_scoped_service('scoped_service')
            instance3 = provider.get_service('scoped_service')
            
            assert instance3['instance'] == 2
    
    def test_non_scoped_service_not_cached(self, service_container):
        """Test that non-scoped services are not cached in ServiceProvider."""
        call_count = [0]
        
        def factory():
            call_count[0] += 1
            return {"instance": call_count[0]}
        
        service_container.register_transient('transient_service', factory)
        
        with ServiceProvider(service_container) as provider:
            # Don't mark as scoped
            instance1 = provider.get_service('transient_service')
            instance2 = provider.get_service('transient_service')
            
            # Should create new instances each time
            assert instance1['instance'] == 1
            assert instance2['instance'] == 2
            assert instance1 is not instance2
    
    def test_scope_cleanup(self, service_container):
        """Test that scoped instances are cleaned up after scope exits."""
        service_container.register_transient('scoped_service', lambda: {"value": 42})
        
        provider = ServiceProvider(service_container)
        provider.register_scoped_service('scoped_service')
        
        with provider:
            instance = provider.get_service('scoped_service')
            assert instance['value'] == 42
            assert len(provider._scoped_instances) == 1
        
        # After exiting scope, instances should be cleaned up
        assert len(provider._scoped_instances) == 0
    
    def test_mixed_service_lifetimes(self, service_container):
        """Test mixing singleton, transient, and scoped services."""
        # Register different lifetime services
        service_container.register_singleton('singleton', {"type": "singleton"})
        service_container.register_transient('transient', lambda: {"type": "transient"})
        service_container.register_transient('scoped', lambda: {"type": "scoped"})
        
        with ServiceProvider(service_container) as provider:
            provider.register_scoped_service('scoped')
            
            # Singleton should be same instance
            singleton1 = provider.get_service('singleton')
            singleton2 = provider.get_service('singleton')
            assert singleton1 is singleton2
            
            # Transient should be different instances
            transient1 = provider.get_service('transient')
            transient2 = provider.get_service('transient')
            assert transient1 is not transient2
            
            # Scoped should be same instance within scope
            scoped1 = provider.get_service('scoped')
            scoped2 = provider.get_service('scoped')
            assert scoped1 is scoped2
    
    def test_service_provider_rejects_unregistered_service(self, service_container):
        """Test that ServiceProvider rejects unregistered services."""
        with ServiceProvider(service_container) as provider:
            with pytest.raises(ValueError, match="Service 'nonexistent' is not registered"):
                provider.get_service('nonexistent')
