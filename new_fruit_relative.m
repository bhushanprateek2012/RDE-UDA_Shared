clc;
clear all;
addpath('./fruits/');
disp('Loading data...');
logFile = strcat('console_log_', datestr(now, 'yyyymmdd_HHMMSS'), '.txt');
diary(logFile);
diary on;
% Open file for writing results
fileID = fopen('fruit_data.csv', 'w');
dataset = {'FruitsO','FruitsP' };

% Parameter initialization
%'eta', 10^(-1), 1 means CD, 0 means MMD
params = struct('delta', 0.9, 'NN', 10, 'k', 100, 'eta', .1, 'sigma', 2, ...
                'ck', 600, 'no_reps', 30);
manifold = struct('k', params.NN, 'Metric', 'Cosine', 'NeighborMode', 'KNN', ...
                  'WeightMode', 'Cosine');

for noIter=[30]
    params.no_reps=noIter;
    for ckIter=[100 200 300]
        params.ck=ckIter;
        for sigmaIter=[20 22 24]
            params.sigma=sigmaIter;
            for etaIter=[.01 .1]
                params.eta=etaIter;
                for kIter=[2 3 4]
                    params.k=kIter;
                    for NNIter=[10]
                        params.NN=NNIter;
                        manifold.k=params.NN;
                        for deltaItr=[.6 .7]
                            params.delta=deltaItr;

                            % Iterate over dataset pairs
                            idx=1;
                            idx1=2;
                            results=[];

                            % Load source and target domain data
                            src = load(char(dataset{idx}));
                            dst = load(char(dataset{idx1}));

                            % Extract features and labels
                            X_src = src.X; X_src=X_src';Y_src = src.Y;
                            X_dst = dst.X; X_dst=X_dst';Y_dst = dst.Y;

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
                            perms_ineq = classperms(C, params.no_reps);
                            relative = genConstraint(Y_src, perms_ineq, []);

                            A0 = computeRelativeLoss(X_src, relative, params.ck);
                            %A0=I;

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
                            Z_src=X_src*W;
                            Z_dst=X_dst*W;

                            % Write maximum accuracy to file
                            acc=max(AC);
                            datasetName=strcat(dataset{idx},'-vs-',dataset{idx1},'_',num2str(acc));
                            fprintf(fileID, '%s %f\n', strcat(dataset{idx}, dataset{idx1}, '.csv'), max(AC));
                            if isempty(results) || acc > max(results)
                                if ~exist('NewResults', 'dir')
                                    mkdir('NewResults');
                                end
                                timestamp = datestr(now, 'yyyymmdd_HHMMSS');
                                saveFile=strcat('NewResults/', datasetName, '_', timestamp, '.mat');
                                save(saveFile, 'X_src', 'X_dst', 'Y_src', 'Y_dst','Z_src','Z_dst','preds','acc', 'datasetName', 'params');
                            end
                            fprintf("idx:%2.0f idx1:%2.0f %s, Acc: %.4f delta: %2.2f NNIter: %2.0f k: %2.0f etaIter: %0.2f sigmaIter: %2.0f ck: %2.0f no_reps: %2.0f\n", ...
                                idx, idx1, datasetName, acc, params.delta, params.NN, params.k, params.eta, params.sigma, params.ck, params.no_reps);
                            results=[results,max(AC)];
                        end
                    end
                end
            end
        end
    end
end

fprintf("EXPERIMENt END");
% Close file
fclose(fileID);
diary off;