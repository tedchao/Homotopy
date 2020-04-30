function A = ladder_matrix(n)
%Generate ladder matrix with n pts
%n should be even

%Construct Tri-diagonal part
for i = 1:n
    A(i,i) = 2*rand(1) - 1;
end

for i = 1:n-1
    A(i,i+1) = 2*rand(1) - 1;
end

for i = 1:n-1
    A(i+1,i) = 2*rand(1) - 1;
end

%Anti-diagonal part
for i = 1:n
    A(n+1-i,i) = 2*rand(1) - 1;
end
