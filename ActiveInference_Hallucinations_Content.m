function MDP = ActiveInference_Hallucinations_Content

% This script is to demonstrate that policy precision and sensory precision
% can provide opposite influences over hidden state inferences. In
% addition, we include content states which can be manipulated to provide
% in or out of context hallucinations 

% In this context what we are doing is having an agent who can listen and
% enter into conversation. So three hidden state factors: listen/notlisten
% and talk/nottalk, and content. This corresponds to three outcomes: hear/nothear and
% proprioceptive outcomes consistent with speaking, or not, as well as
% content. Content is the words of the song "It's a small world after all!"
% Many different kinds of policies will be specified allowing
% for different sequences of listening/notlistening, speaking/notspeaking.
% These will sometimes follow the flow of conversation, sometimes not.

% The mapping A from listening to hearing will have some noise to allow for
% modulation of precision. Initially we will not have any noise in the
% proprioceptive mapping but this can change in the future.

% Crucially, states 1 and 2 are CONTROL STATES and require a B matrix, as both
% listening and speaking (or their negations) are ACTIONS. The content
% state has a B matrix that governs transitions 

rng default
% OPTIONS.precision = 1;
% OPTIONS.plr = 0.1;
% P(s_1)
%--------------------------------------------------------------------------
D{1} = [1 0]'; %STATE 1- (listening and not listening, with equal initial probability)
D{2} = [0 1]';  % STATE 2- (talking and not talking, with equal initial probability)
D{3} = [1 0 0 0 0 0]'; %STATE 3- content, 6 levels of content; these can be sentences. Probability of one to start in the first content state

% P(o|s)
%--------------------------------------------------------------------------

%SENSORY PRECISION SETTING (here are some different values you can try) 
zeta = 1;   %low likelihood precision 
% zeta = 3; %high likelihood precision

for f1 = 1:length(D{1})
    
    for f2 = 1:length(D{2})
        
        for f3 = 1:length(D{3})
            
            if f1 == 2 || f2 == 2 % i.e. not talking or listening; note that the listening case covers the case when someone else should be speaking 
                
                a{1}(7,f1,f2,f3) = 1;  % silence
                
            else
                a{1}(f3,f1,f2,f3)= 1; % hears content
                  

            end
            
        end
        
    end
end

%adding in zeta, correcting for spm_softmax issue 
for f1 = 1:length(D{1})
                for f2 = 1:length(D{2})
                    
               a{1}(:,f1,f2,:) = 512*spm_softmax(zeta*log(a{1}(:,f1,f2,:)+exp(-10)));
                end
             end

for f1 = 1:2
    for f2 = 1:2
        for f3 = 1:6
            if f2 == 1 || f1 == 1
                A{1}(f3,f1,f2,f3) = 1; %GENERATIVE PROCESS (OF THE ENVIRONMENT/OTHER AGENT), SOUND PRESENT OR ABSENT depending on if one of the agents is speaking
                
            else
                A{1}(7,f1,f2,f3) = 1;
                
            end
        end
    end
end

for f1 = 1:length(D{1})
    
    for f2 = 1:length(D{2})
        
        for f3 = 1:length(D{3})
            
            a{2}(f2,f1,f2,f3) = 1;  %proprioception 
            
        end
    end
end

a{2} = 512*a{2};

for f1 = 1:length(D{1})
    
    for f2 = 1:length(D{2})
        
        for f3 = 1:length(D{3})
            
            A{2}(f2,f1,f2,f3) = 1;  %proprioceptive generative process 
            
            
        end
    end
end

% P(s_t+1|s_t,pi)
%--------------------------------------------------------------------------
b{1} = zeros(2); %control over listening in generative model (Agent)
for k = 1:2
    b{1}(:,:,k) = 0;
    b{1}(k,:,k) = 1;
end

b{2} = zeros(2); %control over speaking in generative model
for k = 1:2
    b{2}(:,:,k) = 0;
    b{2}(k,:,k) = 1;
end

b{3}= [0 0 0 0 0 1
       1 0 0 0 0 0
       0 1 0 0 0 0
       0 0 1 0 0 0
       0 0 0 1 0 0
       0 0 0 0 1 0]; %transitions between content states, state 1--> 2---> 3, etc... This is the NORMAL transition matrix for content 


%state (column)/row probability of transition --> this is the ABNORMAL
%transition matrix 
%   b{3}= [0.0 0.0 0.0 0.0 0.0 1.0
%          0.2 0.0 0.0 0.0 0.0 0.0
%          0.0 1.0 0.0 0.0 0.0 0.0
%          0.0 0.0 1.0 0.0 0.0 0.0
%          0.8 0.0 0.0 1.0 0.0 0.0
%          0.0 0.0 0.0 0.0 1.0 0.0];
%    with the anomalous transition matrix, low precision of likelihood allows hallucination 
   
omega = 5;

b{3} = spm_softmax(omega*log(b{3}+exp(-30)));

%
B{1} = zeros(2); %Gnerative process for auditory outcomes- alternating
for k = 1:2
    B{1}(:,:,1) = [0 1;1 0];
    B{1}(:,:,2) = [0 1;1 0];
    
end

B{2} = zeros(2);
for k = 1:2
    B{2}(:,:,k) = 0;
    B{2}(k,:,k) = 1;
end

B{3}= [0 0 0 0 0 1
       1 0 0 0 0 0
       0 1 0 0 0 0
       0 0 1 0 0 0
       0 0 0 1 0 0
       0 0 0 0 1 0]; %content states, generative process (companion agent song order) - shared with generative model (of agent)

% 
% P(o)
%--------------------------------------------------------------------------
C{1} = [0 0 0 0 0 0 0]'; % auditory (sentences 1:6 and silence); prior preferences 
C{2} = [0,0]'; %proprioceptive
% VB inversion
%--------------------------------------------------------------------------
MDP.A = A;
MDP.B = B;
MDP.C = C;
MDP.D = D;
MDP.T = 6;
MDP.a = a;
MDP.b = b;

 MDP.V(:,:,1) = [1 1 1 1 1;2 2 2 2 2;1 2 1 2 1]'; %policies: always listen, always speak, alternate speaking and listening
  MDP.V(:,:,2) = [2 2 2 2 2;1 1 1 1 1;2 1 2 1 2]'; 
  MDP.V(:,:,3) = ones(5,3);

MDP.beta = exp(-2); %Gamma, or policy precision 

mdp = spm_MDP_check(MDP); %check the MDP
clear MDP

MDP = spm_MDP_VB_X(mdp);%generate the MDP


spm_figure('GetWin','Figure 1'); %generate the SPM figure 
spm_MDP_VB_trial(MDP)


% ALL CODE AFTER THIS POINT IS EXTRA CODE NOT NEEDED FOR SIMULATION
% % 
%   MDP.beta = exp(64); %lowered precision
% MDP.beta = exp(-64);
%   MDP.beta = exp(-100); %increased precision
% % b{3}=  0.5*(exp(-10)+[0 0 0 0 0 1
%        1 0 0 0 0 0
%        0 1 0 0 0 0
%        0 0 1 0 0 0
%        0 0 0 1 0 0
%        0 0 0 0 1 0]); %transitions between content states, state 1--> 2---> 3, etc... This is the NORMAL transition matrix for content 
%for manipulating multiple content states 
%   b{3}= [0.0 0.0 0.0 0.0 0.0 1.0
%          1.0 0.0 0.8 0.0 0.0 0.0
%          0.0 1.0 0.0 0.0 0.0 0.0
%          0.0 0.0 0.2 0.0 0.0 0.0
%          0.0 0.0 0.0 1.0 0.0 0.0
%          0.0 0.0 0.0 0.0 1.0 0.0];

% zeta = 0.525;
% zeta = 0.7;
% zeta = 1;
% zeta = 0.3;
% MDP.V(:,:,1) = [1 2 1 2;2 1 1 2;1 2 1 1;2 1 2 2;2 2 2 2;1 1 1 1;1 2 2 1]'; %POLICIES NEED 3 LEVELS NOW
% MDP.V(:,:,2) = [2 1 2 1;1 2 2 1;2 1 2 2;1 2 1 1;1 1 1 1;2 2 2 2;2 1 1 2]';
% MDP.V(:,:,3) = ones(7,4);

% 
% MDP.V(:,:,1) = [2 2 2 2 2 2;1 1 1 1 1 1;1 2 1 2 1 2;1 1 1 2 2 2]'; %POLICIES NEED 3 LEVELS NOW
% MDP.V(:,:,2) = [1 1 1 1 1 1;2 2 2 2 2 2;2 1 2 1 2 1;2 2 2 1 1 1]';
% MDP.V(:,:,3) = ones(6,4);


% % 
% MDP.V(:,:,1) = [1 2 1;2 1 1;2 2 1;2 1 2;2 2 2;1 1 1;1 2 2]'; %POLICIES NEED 3 LEVELS NOW
% MDP.V(:,:,2) = [2 1 2;1 2 2;1 1 2;1 2 1;1 1 1;2 2 2;2 1 1]';
% MDP.V(:,:,3) = ones(3,7);

% MDP.V(:,:,1) = [1 1 1 1 1;2 2 2 2 2;2 1 2 1 2;1 2 1 2 1]'; %POLICIES NEED 3 LEVELS NOW
% MDP.V(:,:,2) = [2 2 2 2 2;1 1 1 1 1;1 2 1 2 1;2 1 2 1 2]';
% MDP.V(:,:,3) = ones(5,4);
% % does not hallucinate but makes a mistake re: when it is speaking and
% will have a content state hallucination at that step 

% MDP.V(:,:,1) = [1 1 1 1 1;2 2 2 2 2;2 1 2 1 2;1 2 2 2 2]'; %POLICIES NEED 3 LEVELS NOW
% MDP.V(:,:,2) = [2 2 2 2 2;1 1 1 1 1;1 2 1 2 1;2 1 1 1 1]';
% MDP.V(:,:,3) = ones(5,4);
% %alternate 4 

% 
% MDP.V(:,:,1) = [1 1 1 1 1;2 2 2 2 2;2 1 2 1 2;1 2 1 2 1;1 2 2 2 2]'; %POLICIES NEED 3 LEVELS NOW
% MDP.V(:,:,2) = [2 2 2 2 2;1 1 1 1 1;1 2 1 2 1;2 1 2 1 2;2 1 1 1 1]';
% MDP.V(:,:,3) = ones(5,5);

% MDP.V(:,:,1) = [1 1 1 1 1;2 2 2 2 2;2 1 2 1 2;1 2 2 2 2]'; %POLICIES NEED 3 LEVELS NOW
% MDP.V(:,:,2) = [2 2 2 2 2;1 1 1 1 1;1 2 1 2 1;2 1 1 1 1]';
% MDP.V(:,:,3) = ones(5,4);


% % 
% MDP.V(:,:,1) = [1 1 1 1 1;2 2 2 2 2;2 1 2 1 2]'; %POLICIES NEED 3 LEVELS NOW
% MDP.V(:,:,2) = [2 2 2 2 2;1 1 1 1 1;1 2 1 2 1]';
% MDP.V(:,:,3) = ones(5,3);
% % %hallucinates at all zetas; just confused below 0.3
% % hallucinates specific content state at low zeta 
% % at high gamma, it starts selecting the 11111 policy but doesn't change
% hallucinations much (just shading) 


% experiments
% 
%   MDP.V(:,:,1) = [1 1 1 1;1 2 1 2;2 2 1 1]'; %ps6 THIS ONE FOR GEN-P dose
%   MDP.V(:,:,2) = [2 2 2 2;2 1 2 1;1 1 2 2]'; %ps6
%   MDP.V(:,:,3) = ones(4,3);
% %   
%   MDP.V(:,:,1) = [1 2 1 2 1;2 2 1 1 1;2 2 1 2 2]'; %ps6 THIS ONE FOR GEN-P dose
%   MDP.V(:,:,2) = [2 1 2 1 2;1 1 2 2 2;2 2 1 1 1]'; %ps6
%   MDP.V(:,:,3) = ones(5,3);
% 
 
%   MDP.V(:,:,1) = [1 1 1 1 1]'; %agent never speaks; use to see what the environment does 
%   MDP.V(:,:,2) = [2 2 2 2 2]'; 
%  MDP.V(:,:,1) = [2 2 2 2 2]'; 
%   MDP.V(:,:,2) = [1 1 1 1 1]'; 
%   MDP.V(:,:,3) = ones(5,1);
%   MDP.V(:,:,3) = ones(5,1);
%   MDP.V(:,:,1) = [1 1 1 1 1;1 1 1 1 1;1 2 1 2 1;2 1 2 1 2]'; 
%   MDP.V(:,:,2) = [2 2 2 2 2;1 1 1 1 1;2 1 2 1 2;2 2 1 2 1]'; 
%   MDP.V(:,:,3) = ones(5,4);


% 
% MDP.V(:,:,1) = [1 1;2 2;2 1]'; 
% MDP.V(:,:,2) = [2 2;1 1;1 2]';
% MDP.V(:,:,3) = ones(2,3);
% %hall, content hall
 
% MDP.V(:,:,1) = [1 1;2 2;2 1;1 2]'; 
% MDP.V(:,:,2) = [2 2;1 1;1 2;2 1]';
% MDP.V(:,:,3) = ones(2,4);
% % % no hallucination

% MDP.V(:,:,1) = [1 1;2 2]'; 
% MDP.V(:,:,2) = [2 2;1 1]';
% MDP.V(:,:,3) = ones(2,2);
% %hall, mild content hall

% MDP.V(:,:,1) = [1 1]'; 
% MDP.V(:,:,2) = [2 2]';
% MDP.V(:,:,3) = ones(2,1);
%------
% MDP.V(:,:,1) = [1 1 1 1 1;2 2 2 2 2;1 2 1 2 1;2 1 2 1 2;1 1 1 2 2;2 2 2 1 1;1 1 1 1 2;2 2 2 2 1;1 2 2 2 2;2 1 1 1 1;1 1 1 1 2;2 2 2 2 1]'; %POLICIES NEED 3 LEVELS NOW
% MDP.V(:,:,2) = [2 2 2 2 2;1 1 1 1 1;2 1 2 1 2;1 2 1 2 1;2 2 2 1 1;1 1 1 2 2;2 2 2 2 1;1 1 1 1 2;2 1 1 1 1;1 2 2 2 2;2 2 2 2 1;1 1 1 1 2]';
% MDP.V(:,:,3) = ones(5,12);

% MDP.V(:,:,1) = [1 1 1 1 1;1 2 1 2 1;2 1 2 1 2;1 1 1 2 2;2 2 2 1 1;1 1 1 1 2;2 1 1 1 1;1 1 1 1 2]'; %POLICIES NEED 3 LEVELS NOW
% MDP.V(:,:,2) = [2 2 2 2 2;2 1 2 1 2;1 2 1 2 1;2 2 2 1 1;1 1 1 2 2;2 2 2 2 1;1 2 2 2 2;2 2 2 2 1]';
% MDP.V(:,:,3) = ones(5,8);

% MDP.V(:,:,1) = [1 1 1 1 1;2 2 2 2 2;1 2 1 2 1;2 1 2 1 2;1 1 1 2 2;2 2 2 1 1;1 1 1 1 2;2 2 2 2 1]'; %POLICIES NEED 3 LEVELS NOW
% MDP.V(:,:,2) = [2 2 2 2 2;1 1 1 1 1;2 1 2 1 2;1 2 1 2 1;2 2 2 1 1;1 1 1 2 2;2 2 2 2 1;1 1 1 1 2]';
% MDP.V(:,:,3) = ones(5,8);
% % 
%  MDP.V(:,:,1) = [1 1 1 1 1;2 2 2 2 2;1 2 1 2 1;2 1 2 1 2]'; %POLICIES NEED 3 LEVELS NOW
%  MDP.V(:,:,2) = [2 2 2 2 2;1 1 1 1 1;2 1 2 1 2;1 2 1 2 1]';
%  MDP.V(:,:,3) = ones(5,4);

% spm_figure('GetWin','Figure 2');
% imagesc(1-MDP.X{3}) 






