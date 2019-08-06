%% Setup
load('AlexNetTrainedFull15.mat');
net = net; % Replace with whatever the above file stores the network in
N = 10000; %Final for testing should be N = 5000;
directory = 'G:\My Drive\Image Rec Project\Data\Manually_Annotated\';
load('EmotionDatasetLabels.mat');
trainingDirectories = trainSubDirectory_filePath;
categories = {'neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt', 'none', 'uncertain', 'non-face'};

%% Load images

inputSize = net.Layers(1).InputSize;
% trainVector = zeros(26244, 1, 1, N);
trainVector = zeros(inputSize(1), inputSize(2), 3, N);
trainLabels = cell(1, N);
    
failures = 0;

for k = 1:N
    while(1)
        try
            index = length(trainingDirectories) - (k + failures); 
            imageTemp = imread(strcat(directory, char(trainingDirectories(index))));
%             resizedImage = imresize(imageTemp, [227 227]);
            resizedImage = imresize(imageTemp, [inputSize(1) inputSize(2)]);
%             [featureVector,hogVisualization] = extractHOGFeatures(resizedImage);
%             trainVector(:, :, :, k) = featureVector(1, :);
            trainVector(:, :, :, k) = resizedImage;
            trainLabels(k) = categories(trainExpression(index) + 1);
            break;
        catch
            failures = failures + 1;
            ['Failures: ' ' ' num2str(failures)]
        end
    end

end

trainLabels = categorical(trainLabels);


%% Classify
[Ypred, scores] = classify(net, trainVector);

%% Create the Confusion Matrix
confMat = zeros(11, 11);
for expect = 1:11
    for ident = 1:11
        val = length(find((categories(expect) == trainLabels) & (categories(ident) == Ypred')));
        confMat(expect, ident) = val;
    end
end

%% Create Probability Matrices
probMat = zeros(100, 11, 11);
emotionProbs = zeros(1, 11);
for c = 1:11
    subsetIndices = find(trainLabels == categories(c));
    subset = scores(subsetIndices, :);
    emotionProbs(c) = size(subset, 1)/10000;
    for classification = 1:11
        setOfVals = subset(:, classification);
        totalSize = length(setOfVals);
        sum = 0;
        for bin = 1:100
            binSize = length(find(setOfVals >= ((bin - 1)*0.01) & setOfVals < (bin * 0.01)));
            probMat(bin, classification, c) = max(binSize/totalSize, 1e-7);
            sum = sum + binSize;
        end
        
    end
end

emotionProbsLayered = repmat(emotionProbs, 100, 1, 11);
binProbs = zeros(100, 11);
clear sum

for bin = 1:100
    items = (scores >= ((bin - 1)*0.01) & scores < (bin * 0.01));
    binProbs(bin, :) = max(sum(items, 1)/N, 1e-7);
end

binProbsLayered = repmat(binProbs, 1, 1, 11);