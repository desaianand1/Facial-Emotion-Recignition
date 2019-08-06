%% Setup
load('AlexNetTrainedFull15.mat');
net = net; % Replace with whatever the above file stores the network in
N = 5000; %Final for testing should be N = 5000;
directory = 'G:\My Drive\Image Rec Project\Data\Manually_Annotated\';
load('EmotionDatasetLabels.mat');
testDirectories = testSubDirectory_filePath;
categories = {'neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt', 'none', 'uncertain', 'non-face'};

%% Load images

inputSize = net.Layers(1).InputSize;
testVector = zeros(inputSize(1), inputSize(2), 3, N);
testLabels = cell(1, N);
    
failures = 0;

for k = 1:N
    while(1)
        try
            index = length(testDirectories) - (k + failures); 
            imageTemp = imread(strcat(directory, char(testDirectories(index))));
%             resizedImage = imresize(imageTemp, [227 227]);
            resizedImage = imresize(imageTemp, [inputSize(1) inputSize(2)]);
%             [featureVector,hogVisualization] = extractHOGFeatures(resizedImage);
%             trainVector(:, :, :, k) = featureVector(1, :);
            testVector(:, :, :, k) = resizedImage;
            testLabels(k) = categories(testExpression(index) + 1);
            break;
        catch
            failures = failures + 1;
            ['Failures: ' ' ' num2str(failures)]
        end
    end

end

testLabels = categorical(testLabels);