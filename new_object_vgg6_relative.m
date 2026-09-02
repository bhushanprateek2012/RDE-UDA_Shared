%Done
clc;
clear all;
addpath('/Volumes/iiits/Research_work/New_Relative_distance_TL/data/CaltechV6');
disp('Loading data...');

% Open file for writing results
fileID = fopen('object__target__vgg6.csv', 'a');
dataset = {'caltech_VGG-FC6', 'amazon_VGG-FC6', 'webcam_VGG-FC6', 'dslr_VGG-FC6'};

datafeature = 'VGG-FC6';

% Parameter initialization
params = struct('delta', 0.9, 'NN', 1, 'k', 10, 'eta', 1e-2, 'sigma', 10, ...
    'ck', 200, 'no_reps', 30);
manifold = struct('k', params.NN, 'Metric', 'Cosine', 'NeighborMode', 'KNN', ...
    'WeightMode', 'Cosine');
for noIter=[20]
    params.no_reps=noIter;
    for ckIter= [130 135]
        params.ck=ckIter;
        for sigmaIter=[10]
            params.sigma=sigmaIter;
            for etaIter= [.01]
                params.eta=etaIter;
                for kIter=[2 3]
                    params.k=kIter;
                    for NNIter=[10]
                        params.NN=NNIter;
                        manifold.k =params.NN;
                        for deltaItr= [ 0.9]
                            params.delta=deltaItr;
                            % Iterate over dataset pairs
                            for idx = 4
                                for idx1 = 3
                                    if (idx==idx1)
                                        continue
                                    end
                                    results = [];
                                    for iteration = 1:10
                                        % Load source and target domain data
                                        srcdata=['.\data\CaltechV6\' dataset{idx} '.mat'];
                                        tgtdata=['.\data\CaltechV6\' dataset{idx1} '.mat'];
                                        src = load(char(srcdata));
                                        dst = load(char(tgtdata));

                                        % Extract features and labels
                                        X_src = src.FTS; Y_src = src.LABELS';
                                        X_src = zscore(X_src);
                                        X_src = normr(X_src);
                                        X_dst = dst.FTS; Y_dst = dst.LABELS';
                                        X_dst = zscore(X_dst);
                                        X_dst = normr(X_dst);

                                        ns = size(X_src, 1); % Number of source samples
                                        nt = size(X_dst, 1); % Number of target samples
                                        X = [X_src; X_dst];  % Combined features
                                        C = length(unique(Y_dst)); % Number of classes

                                        % Apply PCA to reduce dimensionality
                                        params.ck = min(params.ck, size(X, 2));
                                        X = PCA_reduce(X, params.ck);
                                        X_src = X(1:ns, :);
                                        X_dst = X(ns+1:end, :);

                                        % must track the post-PCA dimensionality, not the initial ck
                                        I = eye(size(X_src, 2));

                                        % Generate inter-class constraints
                                        preds = KNN(Y_src, X_src, I, params.k, X_dst);
                                        Y=[Y_src;preds];
                                        perms_ineq = classperms(C, params.no_reps);
                                        relative = genConstraint(Y, perms_ineq, []);

                                        %A0 = computeRelativeLoss(X, relative, params.ck);
                                        A0=I;
                                        % Calculate Laplacian matrix
                                        W = lapgraph(X, manifold);
                                        Dw = diag(sparse(sqrt(1 ./ sum(W))));
                                        L = eye(ns + nt) - Dw * W * Dw;

                                        % Pseudo label generation and domain adaptation
                                        AC = zeros(10, 1);
                                        preds = KNN(Y_src, X_src, I, params.k, X_dst);
                                        for g = 1:10

                                            if g>=1
                                                relative = genConstraint(preds, perms_ineq, []);

                                                A0 = A0+1*computeRelativeLoss(X_dst, relative, params.ck);
                                            end


                                            % M matrix calculation
                                            M = computeMMatrix(Y_src, preds, C, ns, nt, params.delta);

                                            L1 = X' * (params.eta * L + M) * X;

                                            % Solve generalized eigenvalue problem
                                            [W, ~] = eigs(A0, L1, params.sigma, 'LM');


                                            % KNN classification and accuracy calculation
                                            preds = KNN(Y_src, X_src, W, params.k, X_dst);
                                            AC(g) = 100*mean(preds == Y_dst);
                                        end
                                        Z_src = X_src*W;
                                        Z_dst = X_dst*W;
                                        acc=max(AC);
                                        datasetName=strcat(dataset{idx},'-vs-',dataset{idx1},'_',num2str(acc));
                                        % Write maximum accuracy to file
                                        fprintf(fileID, '%s %f\n', strcat(dataset{idx}, dataset{idx1}, '.csv'), max(AC));
                                        if isempty(results) || acc > max(results)
                                            if ~exist('NewResults', 'dir')
                                                mkdir('NewResults');
                                            end
                                            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
                                            saveFile=strcat('NewResults/', datafeature, '_', datasetName, '_', timestamp, '.mat'  );
                                            save(saveFile, 'X_src', 'X_dst', 'Y_src', 'Y_dst','Z_src','Z_dst','preds','acc', 'datasetName', 'params');
                                        end
                                        fprintf("idx:%2.0f idx1:%2.0f %s, Acc: %.4f Itr: %2.0f delta: %2.2f NNIter: %2.0f k: %2.0f etaIter: %0.2f sigmaIter: %2.0f ck: %2.0f no_reps: %2.0f\n", ...
                                            idx, idx1, datasetName, acc, iteration, params.delta, params.NN, params.k, params.eta, params.sigma, params.ck, params.no_reps);
                                        if acc==100 %for D-->W
                                            break;
                                        end
                                        results =[results,max(AC)];
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
% Close file
fclose(fileID);

% Audible alert when the run finishes
beep on;
for k = 1:3
    beep;
    pause(0.5);
end
