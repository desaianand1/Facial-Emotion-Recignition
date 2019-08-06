%% One time setup
clear variables
directory = 'G:\My Drive\Image Rec Project\Data\Manually_Annotated\';
load('EmotionDatasetLabels.mat');
N = 10000;
net = alexnet;
inputSize = net.Layers(1).InputSize;
trainingDirectories = trainSubDirectory_filePath;
validationDirectories = validationSubDirectory_filePath;
categories = {'neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt', 'none', 'uncertain', 'non-face', 'not'};
trainVector = zeros(inputSize(1), inputSize(2), 3, N);
trainLabels = cell(1, N);
failures = 0;

%% Load all sample images
for k =1:N
    while(1)
        try
            index = length(trainingDirectories) - (k + failures);
            imageTemp = imread(strcat(directory, char(trainingDirectories(index))));
            trainVector(:, :, :, k) = imresize(imageTemp, [inputSize(1) inputSize(2)]);
            trainLabels(k) = categories(trainExpression(index) + 1);
            break;
        catch
            failures = failures + 1;
            ['FailuresCat: ' num2str(failures)]
        end
    end
end

trainLabels = categorical(trainLabels);

%% Run each classfier on sample images
classfications = zeros(11, N);
scoreClassifications = zeros(11, N);

for category =1:11
    curCat = categories(category);
    filename = strcat('oneEmotion\', curCat{1}, '.mat');
    load(filename);
    [Ypred, scores] = classify(netTransfer, trainVector);
    if(Ypred(1) == "not" && scores(1,1) >=.5) || (Ypred(1) ~= "not" && scores(1,1) <=.5)
        classfications(category, :) = Ypred;
        scoreClassifications(category, :) = scores(:,2);
        
    else
        classfications(category, :) = Ypred;
        scoreClassifications(category, :) = scores(:,1);
    end

end

classfications = categorical(classfications);

%% Create Probability Matrices
scores = scoreClassifications';
probMat = zeros(100, 11, 11);
for c = 1:11
    subsetIndices = find(trainLabels == categories(c));
    subset = scores(subsetIndices, :);
    for classification = 1:11
        setOfVals = subset(:, classification);
        totalSize = length(setOfVals);
        for bin = 1:100
            binSize = length(find(setOfVals >= ((bin - 1)*0.01) & setOfVals < (bin * 0.01)));
            probMat(bin, classification, c) = max(binSize/totalSize, 1e-5);
            
        end
    end
end

binProbs = zeros(100, 11);

for bin = 1:100
    items = (scores >= ((bin - 1)*0.01) & scores < (bin * 0.01));
    binProbs(bin, :) = max(sum(items, 1)/N, 1e-5);
end

