from flask import Flask, render_template, url_for, flash, redirect, request
import numpy as np
import pandas as pd
import pickle
#from flask_sqlalchemy import SQLALchemy
#from forms import RegistrationForm, LoginForm
app= Flask(__name__)
'''
app.config['SECRET_KEY']='6781628aa0c13be0c667fdfe280ba250'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)
posts=[{
	'author':'satyam',
	'title':'Blog post1',
	'content':'first page',
	'date_posted':'22 April 2019'
},
{
	'author':'saty',
	'title':'Blog post2',
	'content':'second page',
	'date_posted':'24 April 2019'

}]'''


model = pickle.load(open('model.pkl', 'rb'))


#for liver disease
model2 = pickle.load(open('model2.pkl', 'rb'))
sc = pickle.load(open('sc.pkl', 'rb'))


@app.route("/")
@app.route("/home")
def home():
  return render_template('home.html')

@app.route("/prediction")
def prediction():
  return render_template('prediction.html')

@app.route("/about")
def about():
  return render_template('about.html')

@app.route("/instructions")
def instructions():
  return render_template('instructions.html')

@app.route("/precaution")
def precaution():
  return render_template('precaution.html')

@app.route('/heart')
def heart():
    return render_template('heart.html')  

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = [ "age", "trestbps","chol","thalach", "oldpeak", "sex_0",
                       "  sex_1", "cp_0", "cp_1", "cp_2", "cp_3","  fbs_0",
                        "restecg_0","restecg_1","restecg_2","exang_0","exang_1",
                        "slope_0","slope_1","slope_2","ca_0","ca_1","ca_2","thal_1",
                        "thal_2","thal_3"]
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
        
    if output == 1:
        res_val = " Patient is affected by heart disease. Please concern with doctor."
    else:
        res_val = "Congratulations! Patient has no heart disease"
        

    return render_template('heart.html', prediction_text='{}'.format(res_val))    


@app.route('/liver')
def liver():
    return render_template('liver.html')

@app.route('/predict2',methods=['POST'])
def predict2():
    inputs = [float(x) for x in request.form.values()]
    inputs = [np.array(inputs)]
    inputs = sc.transform(inputs)
    output = model2.predict(inputs)
        
    if output < 0.6:
        res_val = "Congratulations! You have no disease"
    else:
        res_val = "Patient is affected by liver disease. Please concern with doctor."
        

    return render_template('liver.html', prediction_text='{}'.format(res_val))  
    


'''@app.route("/register", methods=['GET', 'POST'])
def register():
   form=RegistrationForm() 
   if form.validate_on_submit():
      flash(f'Account created for {form.username.data}!', 'success')
      return redirect(url_for('home')) 
   return render_template('register.html', title='Register', form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    form=LoginForm()
    if form.validate_on_submit():
        if form.email.data == 'admin@blog.com' and form.password.data == 'password':
            flash('You have been logged in!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html', title='Login', form=form)'''
    


if __name__ == "__main__":
     app.run(debug=True)