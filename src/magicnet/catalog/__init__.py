"""Card catalog service: Scryfall dump + versioned card records + embeddings.

See README.md in this package.
"""

from magicnet.catalog.client import CardNotFound, CatalogClient, CatalogError, KeyInfo
from magicnet.catalog.service import CatalogService, create_app
from magicnet.catalog.store import CatalogStore

__all__ = ["CardNotFound", "CatalogClient", "CatalogError", "CatalogService", "CatalogStore", "KeyInfo", "create_app"]
