"""
Service container for dependency injection.
Mimics C#'s IServiceCollection pattern for Python applications.
"""
from typing import Any, Callable, Dict, Optional, Type


class ServiceContainer:
    """
    Service container for dependency injection.
    
    This class provides dependency injection similar to C#'s IServiceCollection,
    allowing registration and resolution of services with singleton and transient lifetimes.
    
    Example:
        >>> container = ServiceContainer()
        >>> container.register_singleton('db', DatabaseConnection())
        >>> container.register_transient('classifier', lambda: ActivityClassifier())
        >>> db = container.resolve('db')
    """
    
    def __init__(self):
        """Initialize the service container."""
        self._services: Dict[str, Callable[[], Any]] = {}
        self._singletons: Dict[str, Any] = {}
    
    def register_singleton(self, service_type: str, instance: Any) -> None:
        """
        Register a singleton service (created once and reused).
        
        Args:
            service_type: Unique identifier for the service
            instance: The service instance to register
            
        Example:
            >>> container.register_singleton('config', Config())
        """
        self._singletons[service_type] = instance
    
    def register_transient(self, service_type: str, factory: Callable[[], Any]) -> None:
        """
        Register a transient service (created each time it's resolved).
        
        Args:
            service_type: Unique identifier for the service
            factory: Function that creates a new service instance
            
        Example:
            >>> container.register_transient('classifier', lambda: ActivityClassifier())
        """
        self._services[service_type] = factory
    
    def register_scoped(self, service_type: str, factory: Callable[[], Any]) -> None:
        """
        Register a scoped service (created once per scope/request).
        Currently implemented as transient - can be extended for true scoped behavior.
        
        Args:
            service_type: Unique identifier for the service
            factory: Function that creates a new service instance
        """
        self._services[service_type] = factory
    
    def resolve(self, service_type: str) -> Any:
        """
        Resolve a service instance.
        
        Args:
            service_type: Unique identifier for the service to resolve
            
        Returns:
            The service instance
            
        Raises:
            ValueError: If the service is not registered
            
        Example:
            >>> db = container.resolve('db')
        """
        # Check singletons first
        if service_type in self._singletons:
            return self._singletons[service_type]
        
        # Check transient services
        if service_type in self._services:
            return self._services[service_type]()
        
        raise ValueError(f"Service '{service_type}' is not registered in the container")
    
    def is_registered(self, service_type: str) -> bool:
        """
        Check if a service is registered.
        
        Args:
            service_type: Service identifier to check
            
        Returns:
            True if the service is registered, False otherwise
        """
        return service_type in self._singletons or service_type in self._services
    
    def clear(self) -> None:
        """Clear all registered services."""
        self._services.clear()
        self._singletons.clear()


class ServiceProvider:
    """
    Context manager for scoped service resolution.
    
    This provides a way to create service scopes similar to C#'s IServiceScope,
    useful for request-scoped dependencies in web applications.
    
    Note: Only services explicitly registered as scoped will be cached within the scope.
    Singletons and transients behave according to their registration lifetime.
    
    Example:
        >>> with ServiceProvider(container) as provider:
        ...     db = provider.get_service('db')
        ...     # Use db within this scope
    """
    
    def __init__(self, container: ServiceContainer):
        """
        Initialize the service provider.
        
        Args:
            container: The service container to use
        """
        self.container = container
        self._scoped_instances: Dict[str, Any] = {}
        self._scoped_services: set = set()  # Track which services are scoped
    
    def register_scoped_service(self, service_type: str) -> None:
        """
        Mark a service as scoped for this provider.
        
        Args:
            service_type: Service identifier
        """
        self._scoped_services.add(service_type)
    
    def __enter__(self) -> 'ServiceProvider':
        """Enter the service scope."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the service scope and clean up scoped instances."""
        self._scoped_instances.clear()
    
    def get_service(self, service_type: str) -> Any:
        """
        Get a service within this scope.
        
        For scoped services, returns the same instance within this scope.
        For singletons and transients, delegates to the container.
        
        Args:
            service_type: Service identifier
            
        Returns:
            The service instance
        """
        # For scoped services, cache within this scope
        if service_type in self._scoped_services:
            if service_type in self._scoped_instances:
                return self._scoped_instances[service_type]
            
            # Resolve and cache for this scope
            instance = self.container.resolve(service_type)
            self._scoped_instances[service_type] = instance
            return instance
        
        # For non-scoped services, delegate to container
        return self.container.resolve(service_type)
