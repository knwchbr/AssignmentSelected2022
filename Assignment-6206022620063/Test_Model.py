import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "fake-news-bert-base-uncased"

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def get_prediction(text, convert_to_label=False):
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    d = {
        0: "reliable",
        1: "fake"
    }
    if convert_to_label:
        return d[int(probs.argmax())]
    else:
        return int(probs.argmax())

real_news = """
Tim Tebow Will Attempt Another Comeback, This Time in Baseball -
The New York Times",Daniel Victor,"If at first you don't
succeed, try a different sport. Tim Tebow, who was a
Heisman quarterback at the University of Florida but was
unable to hold an N. F. L. job, is pursuing a career in Major
League Baseball. <SNIPPED>
"""

print(get_prediction(real_news, convert_to_label=True))

# read the test set
test_df = pd.read_csv("test.csv",encoding='latin1')
# make a copy of the testing set
new_df = test_df.copy()
# add a new column that contains the author, title and article content
new_df["new_text"] = new_df["author"].astype(str) + " : " + new_df["title"].astype(str) + " - " + new_df["text"].astype(str)
# get the prediction of all the test set
new_df["label"] = new_df["new_text"].apply(get_prediction)
# make the submission file
final_df = new_df[["id", "label"]]
final_df.to_csv("submit_final.csv", index=False)