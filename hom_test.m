function hom_test(x0,A1)

%forming A0 with same sparsity pattern but different values
[row0,col0,val0] = find(A1);
A_e = sparse(row0,col0,val0*(1e-4)*rand(1));
A_e = full(A_e);

A0 = A1 - A_e;
A0 = [A1(:,1:99) zeros(100,1)];
%A0 = spones(A1);
%solution to start: ones(size(x0,1),1)
b0 = p(ones(size(x0,1),1),A0);

%solution to target: ones(size(x0,1),1)*2
x = ones(size(x0,1),1)*2;
x((size(x0,1)+1)/2:size(x0,1),1) = x((size(x0,1)+1)/2:size(x0,1),1) + 1;
b1 = p(x,A1);

norm(b1-p(x0,A1))

ind = 1;
for t = 0:0.05:1
    H = (1-t)*p(x0,A0) + t*p(x0,A1);
    b = (1-t)*b0 + t*b1;
    A = (1-t)*A0 + t*A1;

    time = 1;
    while norm(H-b) > 1e-12
        H_0 = (1-t)*p(x0,A0) + t*p(x0,A1); %before
        J = Ja(x0,A);
        %r = [2*eye(size(J,2)); zeros(size(J,2),1)'];
        s = (J)\(-H_0+b);
        alpha = ls(x0,A,b,s,0.5);
        x0 = x0 + alpha*s;
        H = (1-t)*p(x0,A0) + t*p(x0,A1); %after
        
        time = time + 1;
        
        if abs(norm(H_0-b)-norm(H-b)) < 1e-12
            break
        end
    end

    H = (1-t)*p(x0,A0) + t*p(x0,A1);
    norm(H-b)
    f(ind) = norm(H-b);
    ind = ind + 1;
end
norm(b1-p(x0,A1))
semilogy(linspace(0,1,21),f,'-o')
xlabel('t');
ylabel('residual norm');

% Line search based on backtracking
% s is direction, l is contraction factor
function a = ls(x,A,b,s,l)
alpha = 1;
while 1
    if norm(p(x+alpha*s,A)-b) <= norm(p(x,A)-b + (1e-4)*alpha*Ja(x,A)*s)
        break
    else
        alpha = alpha*l;
    end
end
a = alpha;

% Exact Jacobian matrix
% Jacobian has a form:
% |J1|J2|
% |J3|J4|
% Size of Jacobian is 2nx2n
function J = Ja(x,A)
n = (size(x,1)+1)/2;
x(2*n) = 0;
%forming J1
for i = 1:n
    for j = 1:n
        J1(i,j) = x(i)*A(i,j);
    end
end
for i = 1:n
    J1(i,i) = J1(i,i) + dot(A(i,:),x(1:n,:));
end

%forming J2
for i = 1:n
    for j = 1:n
        J2(i,j) = x(i+n)*A(i,j);
    end
end
for i = 1:n
    J2(i,i) = J2(i,i) + dot(A(i,:),x(n+1:2*n,:));
end

%forming J3
for i = 1:n
    for j = 1:n
        J3(i,j) = x(i+n)*A(i,j);
    end
end
for i = 1:n
    J3(i,i) = J3(i,i) - dot(A(i,:),x(n+1:2*n,:));
end

%forming J4
for i = 1:n
    for j = 1:n
        J4(i,j) = -x(i)*A(i,j);
    end
end
for i = 1:n
    J4(i,i) = J4(i,i) + dot(A(i,:),x(1:n,:));
end

J = [J1 J2; J3 J4];
J = J(:,1:2*n-1);

% Power flow equation 
function out = p(x,A)
%initial guess size 2n-1
n = size(x,1);
real_n = (n+1)/2;

%forming on diagonal
for i = 1:real_n
    Z(i,i) = x(i);
end
for j = real_n+1:n
    Z(j-real_n,j-real_n) = Z(j-real_n,j-real_n) + x(j)*1i;
end

%plug in nonlinear eqns
P = Z*A*conj(diag(Z));

%squeeze out the output
R = real(P);
I = imag(P);

out = zeros(n+1,1);
ind = size(A,1);
for i = 1:ind
    out(i) = R(i);
end
for i = ind+1:n+1
    out(i) = I(i-ind);
end