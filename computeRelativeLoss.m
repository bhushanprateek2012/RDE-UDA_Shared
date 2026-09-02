function A0 = computeRelativeLoss(X_src, relative, ck)
    % Initialize A0
    A0 = eye(ck);

    % Calculate relative loss using vectorized operations
    for i = 1:length(relative)
        % Get index pairs
        idx1 = relative(i, 1);
        idx2 = relative(i, 2);
        idx3 = relative(i, 3);
        
        % Calculate difference vectors
        diff_ij = X_src(idx1, :) - X_src(idx2, :);
        diff_ik = X_src(idx1, :) - X_src(idx3, :);
        
        % Update A0 using outer product of difference vectors
        A0 = A0 + (diff_ij' * diff_ij) - 1*(diff_ik' * diff_ik);
    end
end