"""Create SQLAlchemy engines from project database settings."""

from sqlalchemy import create_engine


def make_engine(settings):
    return create_engine(
        f"postgresql+psycopg2://"
        f"{settings['pg_user']}:"
        f"{settings['pg_password']}@"
        f"{settings['pg_host']}:"
        f"{settings['pg_port']}/"
        f"{settings['pg_db']}",
    )
