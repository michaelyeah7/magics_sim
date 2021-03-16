function flag_contact = DetectContact(model, q, qdot, contact_cond)
%DetectContact - Detect contact by position and velocity of foot endpoint 
%
% Syntax: flag_contact = DetectContact(q, qdot)
%
% flag_contact: contact flag, 0-uncontact, 1-contact, 2-impact
    global ip;


    flag_contact = zeros(model.NC, 1);

    endpos = zeros(3, model.NC);
    endvel = zeros(3, model.NC);
    for i=1:model.NC
        % Calcualte pos and vel of foot endpoint, column vector
        endpos(:, i) = CalcBodyToBaseCoordinates(model, q, model.idcontact(i), model.contactpoint{i});
        endvel(:, i) = CalcPointVelocity(model, q, qdot, model.idcontact(i), model.contactpoint{i});
        
        % Detect contact
        flag_contact(i) = DeterminContactType(endpos(:, i), endvel(:, i), contact_cond);
    end
%     [endpos(:,1)',endvel(:,1)', flag_contact(1), 99 , endpos(:,2)',endvel(:,2)', flag_contact(2)]

    %% Determin the contact type 
    function flag = DeterminContactType(pos, vel, contact_cond)
        if pos(3) < contact_cond.contact_pos_lb(3)
            if vel(3) < contact_cond.contact_vel_lb(3)
                flag = 2; % impact
            else 
                if vel(3) > contact_cond.contact_vel_ub(3)
                    flag = 0; % uncontact
                else
                    flag = 1; % contact 
                end
            end
        else
            flag = 0; % uncontact
        end
    end
    
end