%done
clc;
clear;close all;
datapath = 'data/';
fileID = fopen('p_object_data.csv','a');
% Log all console output to a timestamped file
logFile = strcat('console_log_', datestr(now, 'yyyymmdd_HHMMSS'), '.txt');
diary(logFile);
diary on;
tic
for ii=1

    params = struct('delta', .2, 'NN', 10, 'k', 50, 'eta', 10^(-1), 'sigma', 5, ...
        'ck', 100, 'no_reps', 10);
    manifold = struct('k', params.NN, 'Metric', 'Cosine', 'NeighborMode', 'KNN', ...
        'WeightMode', 'Cosine');


    srcStrSURF12 = {'Caltech10','Caltech10','Caltech10','amazon',   'amazon','amazon','webcam',  'webcam', 'webcam','dslr',    'dslr',   'dslr'};
    tgtStrSURF12 = {'amazon',   'webcam',   'dslr',     'Caltech10','webcam','dslr',  'Caltech10','amazon','dslr',  'Caltech10','amazon','webcam'};

    srcStrDecaf12 = {'caltech','caltech','caltech','amazon','amazon','amazon','webcam','webcam','webcam','dslr','dslr','dslr'};
    tgtStrDecaf12 = {'amazon','webcam','dslr','caltech','webcam','dslr','caltech','amazon','dslr','caltech','amazon','webcam'};


    datafeature = 'SURF12';
    %datafeature = 'Decaf12';

    idx='object';
    for noIter=[20]
        params.no_reps=noIter;
        for ckIter= [100 110 120 130]
            params.ck=ckIter;
            for sigmaIter=[10]
                params.sigma=sigmaIter;
                for etaIter= [.1]
                    params.eta=etaIter;
                    for kIter=[35 50 60]
                        params.k=kIter;
                        for NNIter=[10]
                            params.NN=NNIter;
                            manifold.k =params.NN;
                            for deltaItr= [ 0.9 1.0]
                                params.delta=deltaItr;
                                for iData =[5]
                                    % for iData = 9
                                    results = [];
                                    for iteration = 1:10

                                        if strcmp(datafeature,'SURF12')
                                            src = char(srcStrSURF12{iData});
                                            tgt = char(tgtStrSURF12{iData});
                                            options.data = strcat(src,'-vs-',tgt);
                                            %fprintf('Data=%s \n',options.data);


                                            % load and preprocess data
                                            load([datapath 'GFKdata/' src '_SURF_L10.mat']);
                                            X_src = fts ./ repmat(sum(fts,2),1,size(fts,2));
                                            Y_src = labels;
                                            X_src = zscore(X_src);
                                            X_src = normr(X_src)';
                                            X_src = X_src';

                                            load([datapath 'GFKdata/' tgt '_SURF_L10.mat']);
                                            X_dst = fts ./ repmat(sum(fts,2),1,size(fts,2));
                                            Y_dst = labels;
                                            X_dst = zscore(X_dst);
                                            X_dst = normr(X_dst)';
                                            X_dst = X_dst';

                                        elseif strcmp(datafeature,'Decaf12')

                                            src = char(srcStrDecaf12{iData});
                                            tgt = char(tgtStrDecaf12{iData});
                                            options.data = strcat(src,'-vs-',tgt);
                                            %fprintf('Data=%s \n',options.data);

                                            % load and preprocess data
                                            data_file = [datapath 'office_caltech_dl_ms0.mat'];
                                            load(data_file, 'ms_data');

                                            sf = strcmp(ms_data.domain_name,src);
                                            src_dm = find(sf);
                                            disp(' ');
                                            disp('src dm: ');
                                            src_data = select_dm_data(ms_data, src_dm);
                                            if isempty(src_data)
                                                disp(['warning: no src data, domain lbl : ' num2str(src_dm)]);k
                                            end
                                            tf = strcmp(ms_data.domain_name,tgt);
                                            tgt_dm = find(tf);
                                            disp(' ');
                                            disp('tgt dm: ');
                                            tgt_data = select_dm_data(ms_data, tgt_dm);
                                            if isempty(tgt_data)
                                                disp(['warning: no tgt data, domain lbl : ' num2st0r(tgt_dm)]);
                                            end

                                            X_src = src_data.ftr;
                                            Y_src = src_data.lbl;
                                            X_src= normr(X_src);


                                            X_dst = tgt_data.ftr;
                                            Y_dst = tgt_data.lbl;
                                            X_dst = normr(X_dst);


                                        end
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
                                        %T_v=cov(X_dst);
                                        %A0=A0+0.01*T_v;

                                        % Calculate Laplacian matrix
                                        W = lapgraph(X, manifold);
                                        Dw = diag(sparse(sqrt(1 ./ sum(W))));
                                        L = eye(ns + nt) - Dw * W * Dw;

                                        % Pseudo label generation and domain adaptation
                                        AC = zeros(10, 1);
                                        preds = KNN(Y_src, X_src, I, params.k, X_dst);
                                        for g = 1:20


                                            if g>=11
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
                                        acc= max(AC);

                                        %save('Results/Fruits_PO_WT_82_65.mat', 'X_src', 'X_dst', 'Y_src', 'Y_dst','Z_src','Z_dst','preds');
                                        % Only save if current accuracy exceeds all previously stored accuracies
                                        datasetName=strcat(src,'-vs-',tgt,'_',num2str(acc));
                                        % Write maximum accuracy to file
                                        fprintf(fileID, '%s %f\n', datasetName, max(AC));
                                        if isempty(results) || acc > max(results)
                                            if ~exist('NewResults', 'dir')
                                                mkdir('NewResults');
                                            end
                                            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
                                            saveFile=strcat('NewResults/', datafeature, '_', datasetName, '_', timestamp, '.mat'  );
                                            save(saveFile, 'X_src', 'X_dst', 'Y_src', 'Y_dst','Z_src','Z_dst','preds','acc', 'datasetName', 'params');

                                        end
                                        fprintf("iData: %.2f::%s, Acc: %.4f Itr: %2.0f delta: %2.2f NNIter: %2.0f k: %2.0f etaIter: %0.2f sigmaIter: %2.0f ck: %2.0f no_reps: %2.0f\n", ...
                                            iData, datasetName, acc, iteration, params.delta, params.NN, params.k, params.eta, params.sigma, params.ck, params.no_reps);

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
    %nbytes = fprintf(fileID,'%f \n',results)
end
fclose(fileID)
toc
diary off