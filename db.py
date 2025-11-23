# db.py
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Table, MetaData, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

# Use DATABASE_URL env var if provided (Postgres), otherwise default to SQLite file 'data/tracking.db'
DATABASE_URL = os.getenv("DATABASE_URL")  # Render Postgres will set this if you create a DB
if DATABASE_URL:
    engine = create_engine(DATABASE_URL, echo=False)
else:
    os.makedirs("data", exist_ok=True)
    engine = create_engine(f"sqlite:///data/tracking.db", echo=False, connect_args={"check_same_thread": False})

metadata = MetaData()

fetched_posts = Table(
    "fetched_posts",
    metadata,
    Column("id", String, primary_key=True),
    Column("subreddit", String, index=True),
    Column("created_utc", DateTime),
    Column("fetched_at", DateTime, default=datetime.utcnow),
)

# create table(s)
metadata.create_all(engine)

Session = sessionmaker(bind=engine)

def mark_post(post_id: str, subreddit: str, created_utc: datetime):
    """Insert or update a post record (id primary key)."""
    sess = Session()
    try:
        stmt = fetched_posts.insert().values(id=post_id, subreddit=subreddit, created_utc=created_utc, fetched_at=datetime.utcnow())
        sess.execute(stmt)
        sess.commit()
    except Exception:
        sess.rollback()
        # If insert fails because already exists, do an update with fetched_at
        try:
            stmt_upd = fetched_posts.update().where(fetched_posts.c.id == post_id).values(fetched_at=datetime.utcnow())
            sess.execute(stmt_upd)
            sess.commit()
        except Exception as e2:
            sess.rollback()
    finally:
        sess.close()

def get_last_time_for_subreddit(subreddit: str):
    """Return the latest created_utc for posts already fetched for subreddit (datetime or None)."""
    sess = Session()
    try:
        q = select(fetched_posts.c.created_utc).where(fetched_posts.c.subreddit == subreddit).order_by(fetched_posts.c.created_utc.desc()).limit(1)
        res = sess.execute(q).scalar()
        return res
    except Exception:
        return None
    finally:
        sess.close()

def post_exists(post_id: str):
    sess = Session()
    try:
        q = select(fetched_posts.c.id).where(fetched_posts.c.id == post_id).limit(1)
        res = sess.execute(q).scalar()
        return bool(res)
    except Exception:
        return False
    finally:
        sess.close()
