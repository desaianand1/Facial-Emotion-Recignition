%% Load data to classify and filepaths
clear variables
load('testingData.mat');
directory = 'G:\My Drive\Image Rec Project\Code and results\oneEmotion\';
load([directory 'oneEmotionProbMat.mat']);

singleEmotionScores = zeros(length(testLabels), 11);
categories = {'neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt', 'none', 'uncertain', 'non-face'};

%% Computing Class Probabilities
for k = 1:11
    load([directory char(categories(k)) '.mat']);
    [Ypred, scores] = classify(netTransfer, testVector);
    option = 1;
    if (strcmp(char(Ypred(1)), 'not') && (scores(1) > scores(2)))
        option = 2;
    end
    singleEmotionScores(:, k) = scores(:, option);
end

%% Calculating Conditional Probabilities
bayesianScores = ones(length(testLabels), 11);
emotionProbs = [80276 146198 29487 16288 8191 5264 28130 5135 35322 13163 88895];
emotionProbs = emotionProbs/sum(emotionProbs);
weights = [1 1 1 1 1 1 1 1 1 1 1];

indices = ceil(100*singleEmotionScores);
for im = 1:length(testLabels)
    for testCat = 1:11
        for catVal = 1:11
            bayesianScores(im, testCat) = bayesianScores(im, testCat) * ...
                probMat(indices(im, catVal), catVal, testCat);% * emotionProbs(testCat);
        end
        bayesianScores(im, testCat) = bayesianScores(im, testCat) * weights(testCat);
    end
end

%% Classifying
classifications = cell(1, length(testLabels));
for im = 1:length(testLabels)
    options = bayesianScores(im, :);
    index = find(options == max(options));
    classifications(im) = categories(index);
end
classifications = categorical(classifications);

%% Create Confusion Matrix
confMat = zeros(11, 11);
for expect = 1:11
    for ident = 1:11
        val = length(find((categories(expect) == testLabels) & (categories(ident) == classifications)));
        confMat(expect, ident) = val;
    end
end