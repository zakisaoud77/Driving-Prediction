"""empty message

Revision ID: 09f608dff0d5
Revises: 7d7734059369
Create Date: 2024-09-10 14:19:57.062775

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '09f608dff0d5'
down_revision = '7d7734059369'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.add_column(sa.Column('address', sa.String(), nullable=True))
        batch_op.drop_column('adress')

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.add_column(sa.Column('adress', sa.VARCHAR(), autoincrement=False, nullable=True))
        batch_op.drop_column('address')

    # ### end Alembic commands ###
