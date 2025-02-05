"""empty message

Revision ID: 377041e9c7a8
Revises: 3e06713bc1b5
Create Date: 2024-09-10 12:15:17.870489

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '377041e9c7a8'
down_revision = '3e06713bc1b5'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('cars')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('cars',
    sa.Column('id', sa.INTEGER(), autoincrement=True, nullable=False),
    sa.Column('name', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('model', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('doors', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.PrimaryKeyConstraint('id', name='cars_pkey')
    )
    # ### end Alembic commands ###
