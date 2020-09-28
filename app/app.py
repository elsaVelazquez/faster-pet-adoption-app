from flask import Flask, render_template, request, jsonify
import pickle
from predict_one import TextClassifier, SentimAnalysis

#instantiate the class
clf = TextClassifier()
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
    data = str(request.form['article_body']) #user input
    pred = str(clf.predict_one([data])) 
    # tf_adopted = (clf.tfidf_adopted([data])) 
    #cosim
    tf_adopted_ = (clf.tfidf_adopted([data])) 
    tf_adopted_str = str(tf_adopted_['cos_sim'][0:])
    tf_adopted = tf_adopted_str[1:10]
    
    tf_adoptable_ = (clf.tfidf_adoptable([data])) 
    tf_adoptable_str = str(tf_adoptable_['cos_sim'][1:])
    tf_adoptable = tf_adoptable_str[1:10]
    sentim = (my_sentim.sentiment_([data])) 
    # if tf_adoptable > tf_adopted:
    #     tf_adoptable = "change description"
    if float(tf_adoptable) > float(tf_adopted):
        rec = "Change the description."
    elif (float(tf_adopted) >  float(tf_adoptable)):
        rec = "Keep the description."
    else:
        "Please check your input and try again."
        
    
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