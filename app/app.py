from flask import Flask, render_template, request, jsonify
import pickle
from predict_one import TextClassifierAdopted, TextClassifierAdoptable, SentimAnalysis

#instantiate the class
clf_adopted = TextClassifierAdopted()
clf_adoptable = TextClassifierAdoptable()
my_sentim = SentimAnalysis()

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    """Render a simple splash page."""
    return render_template('submit.html')

@app.route('/submit', methods=['GET'])
def submit():
    """Render a page containing a textarea input where the user can paste an
    description to be classified.  """
    return render_template('submit.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Recieve the description to be classified from an input form and use the
    models to classify and predict.
    """
    # tf_adopted_ = ''
    # tf_adopted = ''
    # tf_adopted_str = ''
    # tf_adoptable_ = ''
    # tf_adoptable = ''
    # tf_adoptable_str = ''   
    
    
    data = str(request.form['article_body']) #user input
    pred = str(clf_adopted.predict_one([data])) 
    tf_adopted = (clf_adopted.tfidf_adopted([data])) 
    tf_adoptable = (clf_adoptable.tfidf_adoptable([data])) 
    rec = ''
    if float(tf_adoptable) > float(tf_adopted):
        rec = "Change the description."
    elif (float(tf_adopted) >  float(tf_adoptable)):
        rec = "Keep the description."
    else:
        "Please check your input and try again."
    
    #limit how many digits get printed
    # tf_adopted_str = str(tf_adopted)
    # tf_adopted = tf_adopted_str[:6]
    # tf_adoptable_str = str(tf_adoptable)
    # tf_adoptable = tf_adoptable_str[:6]   
    sentim = (my_sentim.sentiment_([data])) 

    return render_template('predict.html', description=data, predicted=pred, cosim_adopted=tf_adopted, cosim_adoptable=tf_adoptable, sentiment=sentim, recommend = rec)

@app.route('/about', methods=['GET'])
def about():
    """More about this project."""
    return render_template('about.html')

@app.route('/contact', methods=['GET'])
def contact():
    """Contact Elsa."""
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5081, debug=True)