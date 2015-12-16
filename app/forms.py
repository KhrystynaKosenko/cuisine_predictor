from flask.ext.wtf import Form
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from wtforms.widgets import TextArea

class IngredientsArea(Form):
    body = StringField(u'Ingredints', validators=[DataRequired()], widget=TextArea())
    submit = SubmitField('Analize')
    random = SubmitField('Random Ingredints')
