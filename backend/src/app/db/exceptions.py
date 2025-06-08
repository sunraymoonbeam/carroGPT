class CollectionNotFoundError(Exception):
    """Raised when a requested Qdrant collection does not exist."""

    pass


class CollectionAlreadyExistsError(Exception):
    """Raised when attempting to create a collection that already exists."""

    pass


class DocumentProcessingError(Exception):
    """
    Raised when a PDF/DOCX/URL cannot be processed or embedded.
    This may include file read/write errors, loader failures, or Qdrant upsert errors.
    """

    pass
