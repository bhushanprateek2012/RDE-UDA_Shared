
clear all;
clc;
addpath('./data/');
disp('Loading data...');

% Open file for writing results
fileID = fopen('coil_data.csv', 'w');
dataset = {'COIL_1', 'COIL_2'};

% Parameter initialization
params = struct('delta', 0.9, 'NN', 1, 'k', 2, 'eta', 10^(-0), 'sigma', 30, ...
                'ck', 180, 'no_reps', 1);
manifold = struct('k', params.NN, 'Metric', 'Cosine', 'NeighborMode', 'KNN', ...
                  'WeightMode', 'Cosine');

% Identity matrix
I = eye(params.ck);

% Iterate over dataset pairs

    
        idx=1;
        idx1=2;
    
        % Load source and target domain data
        src = load(char(dataset{idx}));
        dst = load(char(dataset{idx1}));
        

        % Extract features and labels
        X_src = src.X_src'; Y_src = src.Y_src;
        X_dst = src.X_tar'; Y_dst = src.Y_tar;

        ns = size(X_src, 1); % Number of source samples
        nt = size(X_dst, 1); % Number of target samples
        X = [X_src; X_dst];  % Combined features
        C = length(unique(Y_dst)); % Number of classes

        % Apply PCA to reduce dimensionality
        X = PCA_reduce(X, params.ck);
        X_src = X(1:ns, :);
        X_dst = X(ns+1:end, :);

        % Generate inter-class constraints
        perms_ineq = classperms(C, params.no_reps);
        relative = genConstraint(Y_src, perms_ineq, []);

       A0 = computeRelativeLoss(X_src, relative, params.ck);
        
        T_v=cov(X_dst);
        A0=A0+0.01*T_v;

        % Calculate Laplacian matrix
        W = lapgraph(X, manifold);
        Dw = diag(sparse(sqrt(1 ./ sum(W))));
        L = eye(ns + nt) - Dw * W * Dw;

        % Pseudo label generation and domain adaptation
        AC = zeros(10, 1);
        preds = KNN(Y_src, X_src, I, params.k, X_dst);
        for g = 1:20 

            
            if g>=1
                relative = genConstraint(preds, perms_ineq, []);

       A0 = A0+1*computeRelativeLoss(X_dst, relative, params.ck);
            end
            
            
            % M matrix calculation
            e = [1/ns * ones(ns, 1); -1/nt * ones(nt, 1)];
            M = e * e' * C;

            N = 0;
            for c = unique(Y_src)'
                e = zeros(ns + nt, 1);
                e(Y_src == c) = 1 / sum(Y_src == c);
                e(ns + find(preds == c)) = -1 / sum(preds == c);
                N = N + e * e';
            end

            M = (1 - params.delta) * M + params.delta * N;
            M = M / norm(M, 'fro');
            L1 = X' * (params.eta * L + M) * X;

            % Solve generalized eigenvalue problem
            [W, ~] = eigs(A0, L1, params.sigma, 'LM');
           

            % KNN classification and accuracy calculation
            preds = KNN(Y_src, X_src, W, params.k, X_dst);
            AC(g) = mean(preds == Y_dst);
           
        end
        Z_src=X_src*W;
        Z_dst=X_dst*W;
        % Write maximum accuracy to file
        fprintf(fileID, '%s %f\n', strcat(dataset{idx}, 'vs', dataset{idx1},'.csv'), max(AC));
path = ['NewResults/', dataset{idx}, '.mat'];
accuracy=max(AC)
strcat(dataset{idx}, 'vs', dataset{idx1})
        save(path, 'X_src', 'X_dst', 'Z_src','Z_dst','Y_src', 'Y_dst','preds','accuracy');


% Close file
fclose(fileID);
