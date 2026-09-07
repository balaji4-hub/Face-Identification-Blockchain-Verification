"""
VERIFACE CHAIN Database Connection and Session Management
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from veriface.core.config import settings
from veriface.models.schemas import Base

# Create async engine for SQLite
engine = create_async_engine(
    settings.database_url.replace("sqlite+aiosqlite", "sqlite+aiopg") 
    if "postgresql" in settings.database_url else settings.database_url,
    echo=settings.debug,
    future=True
)

# Session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

# Sync engine for migrations
sync_engine = engine.sync_engine if hasattr(engine, 'sync_engine') else None


async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    """Dependency for FastAPI routes - provides async database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_sync_db():
    """Get sync database session for non-async contexts"""
    SessionLocal = sessionmaker(bind=sync_engine or engine, expire_on_commit=False)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()