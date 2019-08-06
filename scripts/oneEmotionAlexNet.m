%% Testing with AlexNet
clear variables;
directory = 'G:\My Drive\Image Rec Project\Data\Manually_Annotated\';
load('EmotionDatasetLabels.mat');
trainingDirectories = trainSubDirectory_filePath;
validationDirectories = validationSubDirectory_filePath;

allCatsToTest = [1 4 5 6 8 9 10 11];

for category = allCatsToTest
%% Choosing Data Size

%category = 4; %Change to train a different expression
%N is the number of examples of a category
N = 1000; %Final for testing should be N = 2500;
Nvalidate = 500; %Final for testing should be Nvalidate = 1000;


%% Getting Network Information
net = alexnet;
inputSize = net.Layers(1).InputSize;
categories = {'neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt', 'none', 'uncertain', 'non-face', 'not'};

validationVector = zeros(inputSize(1), inputSize(2), 3, Nvalidate);
validationLabels = cell(1, Nvalidate);

failures2 = 0;


%% Validation Setup
CategoryIndexes = find(validationExpression == (category -1));
NonCategoryIndexes = find(validationExpression ~= (category -1));
failuresCat = 0;
failuresNon = 0;

validationVector = zeros(inputSize(1), inputSize(2), 3, 2*Nvalidate);
validationLabels = cell(1, 2*Nvalidate);

for k = 1:Nvalidate
    while(1)
        try
            indexindex = mod((k + failuresCat), length(CategoryIndexes)) + 1;
            index = CategoryIndexes(indexindex);
            imageTemp = imread(strcat(directory, char(validationDirectories(index))));
            validationVector(:, :, :, k) = imresize(imageTemp, [inputSize(1) inputSize(2)]);
            
            validationLabels(k) = categories(category);
            break;
        catch
            failuresCat = failuresCat + 1;
            ['FailuresCatValid: ' num2str(failuresCat)]
        end
    end
    
end


for k = 1:Nvalidate
    while(1)
        try
            indexindex = mod((k + failuresNon), length(NonCategoryIndexes)) + 1;
            index = NonCategoryIndexes(indexindex);
            imageTemp = imread(strcat(directory, char(validationDirectories(index))));
            validationVector(:, :, :, k + Nvalidate) = imresize(imageTemp, [inputSize(1) inputSize(2)]);
            validationLabels(k + Nvalidate) = categories(12);
            
            break;
        catch
            failuresNon = failuresNon + 1;
            ['FailuresNonValid: ' num2str(failuresNon)]
        end
    end
    
end

validationLabels = categorical([validationLabels categories(category) categories(12)]);
validationLabels = validationLabels(1:end-2);


failuresCat = 0;
failuresNon = 0;

layersTransfer = net.Layers(1:end-3);

layers = [layersTransfer; fullyConnectedLayer(2, 'WeightLearnRateFactor', 25, ...
    'BiasLearnRateFactor', 25); softmaxLayer; classificationLayer];

options = trainingOptions('sgdm', 'MiniBatchSize', 35, ...
    'MaxEpochs', 3, 'InitialLearnRate', 1e-4, 'ValidationData', {validationVector, validationLabels}, ...
    'ValidationFrequency', 3, 'ValidationPatience', Inf, ...
    'Verbose', false, 'Plots', 'training-progress');


CategoryIndexes = find(trainExpression == (category -1));
NonCategoryIndexes = find(trainExpression ~= (category -1));


%% Iterative Training
for m = 1:3
    
    clear trainVector trainLabels netTransfer
    trainVector = zeros(inputSize(1), inputSize(2), 3, 2*N);
    trainLabels = cell(1, 2*N);
    
    for k = 1:N
        while(1)
            try
                indexindex = mod((k + (m-1)*N + failuresCat), length(CategoryIndexes)) + 1;
                index = CategoryIndexes(indexindex);
                imageTemp = imread(strcat(directory, char(trainingDirectories(index))));
                trainVector(:, :, :, k) = imresize(imageTemp, [inputSize(1) inputSize(2)]);
                
                trainLabels(k) = categories(category);
                break;
            catch
                failuresCat = failuresCat + 1;
                ['FailuresCat: ' num2str(failuresCat)]
            end
        end
        
    end
    
    
    for k = 1:N
        while(1)
            try
                indexindex = mod((k + (m-1)*N + failuresNon), length(NonCategoryIndexes)) + 1;
                index = NonCategoryIndexes(indexindex);
                imageTemp = imread(strcat(directory, char(trainingDirectories(index))));
                trainVector(:, :, :, k + N) = imresize(imageTemp, [inputSize(1) inputSize(2)]);
                trainLabels(k + N) = categories(12);
                
                break;
            catch
                failuresNon = failuresNon + 1;
                ['FailuresNon: ' num2str(failuresNon)]
            end
        end
        
    end
    
    trainLabels = categorical([trainLabels categories(category) categories(12)]);
    trainLabels = trainLabels(1:end-2);
    
    netTransfer = trainNetwork(trainVector, trainLabels, layers, options);
    %%
    iterativeTrainings(m) = netTransfer;
    layers = netTransfer.Layers;
    
end

%% Saves all open figures and netTransfer
figs = findall(groot, 'Type', 'figure');
curCat = categories(category);
for k = 1:length(figs)
    
    filename = strcat('oneEmotion\TryTwo', curCat{1}, num2str(6-k), '.png');
    saveas(figs(k), filename);
    close(figs(k));
end
filename = strcat('oneEmotion\TryTwo', curCat{1}, '.mat');
save(filename, 'netTransfer');

end

%% Testing
%[Ypred, scores] = classify(net, validationVector);

