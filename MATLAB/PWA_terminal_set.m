clear all
clc

% regions
L = 4; % number of regions
V = [20, 20; -20, 20; 0, 0];
Psi(1) = Polyhedron('V', V);

V = [20, 20; 20, -20; 0, 0];
Psi(2) = Polyhedron('V', V);

V = [20, -20; -20, -20; 0 0];
Psi(3) = Polyhedron('V', V);

V = [-20, -20; -20, 20; 0,0];
Psi(4) = Polyhedron('V', V);

for i=1:L
    Psi(i).plot
    hold on
end
clear A

% dynamics and controller
A{1} = [0.6324, 0.2785; 0.0975, 0.5469];
A{2} = [0.6555, 0.7060; 0.1712, 0.0318];
A{3} = [0.6324, 0.2785; 0.0975, 0.5469];
A{4} = [0.6555, 0.7060; 0.1712, 0.0318];

B{1} = [1; 0];
B{2} = [1; 0];
B{3} = [1; 0];
B{4} = [1; 0];

K{1} = [-0.0544 -0.1398];
K{2} = [-0.1544, -0.0295];
K{3} = [-0.0544 -0.1398];
K{4} = [-0.1544, -0.0295];

u_lim = 3;

for i=1:L
    X_iu(i) = Polyhedron('A', [Psi(i).A; K{i}; -K{i}], 'b', [Psi(i).b; u_lim; u_lim]);
    X_iu(i).plot('color', 'lightblue')
end

%% initial guess for term set
% V = [-20, 20; 3.75, 20; 15.448, 15.488; 20, -2.98; 20, -20; -3.75, -20; -15.448, -15.488; -20, 2.983];
% X_iu_t = Polyhedron('V', V);
g = 47;
P = [7.8514, 8.1971; 8.1957, -7.8503];
A_t = [P; -P];
b_t = g*ones(4, 1);
X_iu_t = Polyhedron('A' , A_t, 'b', b_t);
X_iu_t.plot('color', 'gray');

%% coupling sets
X = Polyhedron('A', [1, 0; -1, 0; 0, 1; 0, -1], 'b', [20;20;20;20]);
A_c = 5e-2*eye(2);
% A_c = 2e-3*eye(2);
A_12 = 2*A_c;
W = A_12*X;

% global A matrix
max_eig = -inf;
for i = 1:L
    for j = 1:L
        for k = 1:L
            A_glob = [A{i}, A_c, zeros(2,2); A_c, A{j}, A_c; A_c, zeros(2,2), A{k}];
            e = abs(eig(A_glob));
            if max(e) > max_eig
                max_eig = max(e);
            end
        end
    end
end
max_eig

for i=1:L
    Phi{i} = A{i} + B{i}*K{i};
end

% global A matrix closed loops
max_eig = -inf;
for i = 1:L
    for j = 1:L
        for k = 1:L
            A_glob_cl = [Phi{i} , A_c, zeros(2,2); A_c, Phi{j}, A_c; A_c, zeros(2,2), Phi{k}];
            e = abs(eig(A_glob_cl));
            if max(e) > max_eig
                max_eig = max(e);
            end
        end
    end
end
max_eig

figure(2)
X_t = X_iu_t;
for r=1:50
    r
    A = [];
    b = [];
    empty_flag = 0;
    temp = minus(X_t, A_12*X_t)
    for i=1:L
        % Q = inv(Phi{i})*X_t;
        % Q = inv(Phi{i})*(minus(X_t, W));
        Q = inv(Phi{i})*(minus(X_t, A_12*X_t));
        if Q.isEmptySet()
            empty_flag = 1;
            break
        else
            X_temp = Polyhedron('A', [Q.A; X_t.A], 'b', [Q.b; X_t.b]);
        end
        A = [A; X_temp.A];
        b = [b; X_temp.b];
    end
    if empty_flag
        X_t = Polyhedron();
        break
    else
        X_t = Polyhedron('A', A, 'b', b);
        X_t = X_t.minHRep();
    end
    X_t.plot
    hold on
    waitforbuttonpress
end
r
X_t
figure(1)
X_t.plot('color', 'pink')
save('X_t.mat', 'X_t')



