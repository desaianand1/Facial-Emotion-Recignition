% % Sample data (54000 x 10)
% data = rand(54000,10);
% % Cross varidation (train: 70%, test: 30%)
% cv = cvpartition(size(data,1),'HoldOut',0.3);
% idx = cv.test;
% % Separate to training and test data
% dataTrain = data(~idx,:);
% dataTest  = data(idx,:);

rng('default');
indices =  randperm(size(face_height,1));

arousal =arousal(indices);
expression =expression(indices);
face_height =face_height(indices);
face_width =face_width(indices);
face_x =face_x(indices);
face_y =face_y(indices);
facial_landmarks= facial_landmarks(indices);
subDirectory_filePath= subDirectory_filePath(indices);
valence= valence(indices);

N = length(valence);
trainRatio = 0.7;
testRatio = 1-trainRatio;

trainValence = valence(1:floor(N*trainRatio));
testValence = valence(floor(N*trainRatio)+1:end);

trainArousal = arousal(1:floor(N*trainRatio));
testArousal = arousal(floor(N*trainRatio)+1:end);

trainExpression = expression(1:floor(N*trainRatio));
testExpression = expression(floor(N*trainRatio)+1:end);

trainFace_height = face_height(1:floor(N*trainRatio));
testFace_height = face_height(floor(N*trainRatio)+1:end);

trainFace_width = face_width(1:floor(N*trainRatio));
testFace_width = face_width(floor(N*trainRatio)+1:end);

trainFace_x = face_x(1:floor(N*trainRatio));
testFace_x = face_x(floor(N*trainRatio)+1:end);

trainFace_y = face_y(1:floor(N*trainRatio));
testFace_y = face_y(floor(N*trainRatio)+1:end);

trainFacial_landmarks = facial_landmarks(1:floor(N*trainRatio));
testFacial_landmarks = facial_landmarks(floor(N*trainRatio)+1:end);

trainSubDirectory_filePath = subDirectory_filePath(1:floor(N*trainRatio));
testSubDirectory_filePath = subDirectory_filePath(floor(N*trainRatio)+1:end);
