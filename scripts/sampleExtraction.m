clear variables
load('SingleScores.mat');
load('..\Gabe Temp Uploads\testingData.mat');

%%
categories = {'neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt', 'none', 'uncertain', 'non-face', 'not'};
classifications = testLabels;
for k = 1:5000
    classifications(k) = categories(find(index(:, k) == 1));
end
correctIndexes = find(testLabels == classifications);
incorrectIndexes = find(testLabels ~= classifications);
correctScores = max(scores(:, correctIndexes));
incorrectScores = max(scores(:, incorrectIndexes));
maxScoresOverall = max(scores);

%% Strong Positives
% Note that section 2 needs to be rerun before each subsequent section
for k = 1:100
    maxVal = max(correctScores);
    maxIndex = find(maxScoresOverall == maxVal);
    figure(k)
    imshow(uint8(testVector(:, :, :, maxIndex)));
    title([char(testLabels(maxIndex)) ': ' num2str(maxVal)]);
    correctScores(find(correctScores == maxVal)) = 0;
end

%% Weak Positives
% Note that section 2 needs to be rerun before each subsequent section
for k = 1:100
    minVal = min(correctScores);
    minIndex = find(maxScoresOverall == minVal);
    figure(k)
    imshow(uint8(testVector(:, :, :, minIndex)));
    title([char(testLabels(minIndex)) ': ' num2str(minVal)]);
    correctScores(find(correctScores == minVal)) = 1;
end

%% Strong Negatives
% Note that section 2 needs to be rerun before each subsequent section
for k = 1:100
    maxVal = max(incorrectScores);
    maxIndex = find(maxScoresOverall == maxVal);
    figure(k)
    imshow(uint8(testVector(:, :, :, maxIndex)));
    title([char(classifications(maxIndex)) ': ' num2str(maxVal) ' Correct:' char(testLabels(maxIndex))]);
    incorrectScores(find(incorrectScores == maxVal)) = 0;
end

%% Weak Negatives
% Note that section 2 needs to be rerun before each subsequent section
for k = 1:100
    minVal = min(incorrectScores);
    minIndex = find(maxScoresOverall == minVal);
    figure(k)
    imshow(uint8(testVector(:, :, :, minIndex)));
    title([char(classifications(minIndex)) ': ' num2str(minVal) ' Correct:' char(testLabels(minIndex))]);
    incorrectScores(find(incorrectScores == minVal)) = 1;
end



