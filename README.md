# Facial Emotion Recignition

An image classifier that can categorize a face into one of the following 11 emotions with an accuracy of 40.91% :

| Neutral | Happy | Sad | Surprise | Fear | Disgust | Anger | Contempt | None | Uncertain | Non-Face |
|---|---|---|---|---|---|---|---|---|---|---|

# Architecture

A pretrained AlexNet neural network was used with all but the last 3 layers transfered over. The last 3 layers were replaced with a fully connected, a softmax and a classification output layer. This net was trained on 450,000 manually annotated images from the AffectNet(wild) database. The data was split into training and test set in a 7:3 ratio. Additionally, since the initial dataset was unbalanced and provided trivial results, the dataset was artificially balanced to account for discrepencies in the number of images for each emotion class.

In total, two reasonably successful classifiers were created:   [Note: Random Guessing accuracy = 9.09%]
1. Multi-class output classifier (accuracy = 12.36%)
2. Single emotion binary output classifier, for each emotion (eg: Angry or Not Angry)
    * combining results from 11 single emotion classifiers, two methods were used to come up with an overall prediction:
      * Highest confidence score: 40.91% 
      * Weighted Bayesian classification : 16.36% 

The highest confidence score from the combined results of the 11 single emotion classifiers resulted in an overall accuracy of **40.91%**

# ROC Curve </br> 
(for all 11 single emotion classifiers on an unbalanced test set)

![alt text](/sample%20results/roc.png "ROC Curve for all 11 single emotion classifiers on an unbalanced test set")

# Sample Results

![alt text](/sample%20results/happy.png "Happy face - confidence: 0.9993")
</br>&nbsp;&nbsp;&nbsp;&nbsp;**Happy Face** (confidence: 0.9993)
</br>
![alt text](/sample%20results/surprise.png "Surprised face - confidence: 0.99804")
</br>&nbsp;&nbsp;&nbsp;&nbsp;**Surprised face** (confidence: 0.99804)
