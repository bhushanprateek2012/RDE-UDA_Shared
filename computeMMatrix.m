% Function to compute M matrix
function M = computeMMatrix(Y_src, preds, C, ns, nt, delta)
    e = [1/ns * ones(ns, 1); -1/nt * ones(nt, 1)];
    M = e * e' * C;

    N = 0;
    for c = unique(Y_src)'
        e = zeros(ns + nt, 1);
        e(Y_src == c) = 1 / sum(Y_src == c);
        e(ns + find(preds == c)) = -1 / sum(preds == c);
        N = N + e * e';
    end

    M = (1 - delta) * M + delta * N;
    M = M / norm(M, 'fro');
end