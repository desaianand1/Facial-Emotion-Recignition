%% Testing with AlexNet
%% Choosing Data Size
N = 200; %Final for testing should be N = 5000;
Nvalidate = 2000; %Final for testing should be Nvalidate = 1000;
directory = 'G:\My Drive\Image Rec Project\Data\Manually_Annotated\';
load('EmotionDatasetLabels.mat');
trainingDirectories = trainSubDirectory_filePath;
validationDirectories = validationSubDirectory_filePath;

%% Getting Network Information
net = alexnet;
inputSize = net.Layers(1).InputSize;
categories = {'neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt', 'none', 'uncertain', 'non-face'};

validationVector = zeros(inputSize(1), inputSize(2), 3, Nvalidate);
validationLabels = cell(1, Nvalidate);

failures2 = 0;

for k = 1:Nvalidate
    while(1)
        try
            imageTemp = imread(strcat(directory, char(validationDirectories(k + failures2))));
            validationVector(:, :, :, k) = imresize(imageTemp, [inputSize(1) inputSize(2)]);
            validationLabels(k) = categories(uint8(validationExpression(k))+1);
            break;
        catch
            failures2 = failures2 + 1;
            ['Failures2: ' num2str(failures2)]
        end
    end
end

validationLabels = categorical([validationLabels categories]);
validationLabels = validationLabels(1:end - 11);


failures = 0;

%% Prepping Network
layersTransfer = net.Layers(1:end-3);

layers = [layersTransfer; fullyConnectedLayer(11, 'WeightLearnRateFactor', 50, ...
    'BiasLearnRateFactor', 50); softmaxLayer; classificationLayer];

options = trainingOptions('sgdm', 'MiniBatchSize', 35, ...
    'MaxEpochs', 3, 'InitialLearnRate', 2e-4, 'ValidationData', {validationVector, validationLabels}, ...
    'ValidationFrequency', 3, 'ValidationPatience', Inf, ...
    'Verbose', false, 'Plots', 'training-progress');




%% Iterative Training

failures = zeros(1, 11);

for m = 1:50
    
    clear trainVector trainLabels netTransfer
    trainVector = zeros(inputSize(1), inputSize(2), 3, 11*N);
    trainLabels = cell(1, 11*N);
    
    for c = 1:11
        
        emotionOptions = find(trainExpression == (c -1));
        

        for k = 1:N
            while(1)
            try
                index = emotionOptions(mod((k + (m-1)*N + failures(c)), length(emotionOptions)) + 1); 
                imageTemp = imread(strcat(directory, char(trainingDirectories(index))));
                trainVector(:, :, :, k) = imresize(imageTemp, [inputSize(1) inputSize(2)]);
                trainLabels(k + N*(c-1)) = categories(c);
                break;
            catch
                failures(c) = failures(c) + 1;
                ['Failures: ' num2str(c) ' ' num2str(failures(c))]
            end
        end

        end
    end

    trainLabels = categorical(trainLabels);
    indices =  randperm(size(trainLabels, 2));
    trainLabels(1:(N*11)) = trainLabels(indices);
    secondTrainVector = trainVector;
    for p = 1:N*11
        secondTrainVector(:, :, :, p) = trainVector(:, :, :, indices(p));
    end
    
    netTransfer = trainNetwork(secondTrainVector, trainLabels, layers, options);
    %%
    iterativeTrainings(m) = netTransfer;
    layers = netTransfer.Layers;

end
%% 


%% Testing
[Ypred, scores] = classify(iterativeTrainings(37), validationVector);

