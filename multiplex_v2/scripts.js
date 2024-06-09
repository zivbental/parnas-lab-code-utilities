$(document).ready(function() {
    $('#startExperiment').click(function() {
        var selectedOption = $('#experiment_option').val();
        $.ajax({
            url: '/start_experiment',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({experiment_option: selectedOption}),
            success: function(response) {
                $('#message').text(response.message);
            },
            error: function(xhr, status, error) {
                $('#message').text("An error occurred: " + xhr.responseText);
            }
        });
    });

    function updateChamberStates() {
        $.ajax({
            url: '/get_chamber_states',
            type: 'GET',
            success: function(response) {
                response.chambers.forEach(function(chamber) {
                    var leftShockBox = $('#chamber-' + chamber.id + ' .left-shock-box');
                    if (chamber.leftShock) {
                        leftShockBox.removeClass('red').addClass('green');
                    } else {
                        leftShockBox.removeClass('green').addClass('red');
                    }

                    var rightShockBox = $('#chamber-' + chamber.id + ' .right-shock-box');
                    if (chamber.rightShock) {
                        rightShockBox.removeClass('red').addClass('green');
                    } else {
                        rightShockBox.removeClass('green').addClass('red');
                    }
                });
            },
            error: function(xhr, status, error) {
                console.error("An error occurred while fetching chamber states: " + xhr.responseText);
            }
        });
    }

    setInterval(updateChamberStates, 1000); // Poll every 1 second
});

function shockLeft() {
    window.location.href = "/shock_left";
}

function removeShockLeft() {
    window.location.href = "/remove_shock_left";
}

function shockRight() {
    window.location.href = "/shock_right";
}

function removeShockRight() {
    window.location.href = "/remove_shock_right";
}

function activateOdorLeft() {
    window.location.href = "/activate_odor_left";
}

function activateOdorRight() {
    window.location.href = "/activate_odor_right";
}

function disableOdorLeft() {
    window.location.href = "/disable_odor_left";
}

function disableOdorRight() {
    window.location.href = "/disable_odor_right";
}

function activateAirflow() {
    window.location.href = "/activate_air_flow";
}

function disableAirflow() {
    window.location.href = "/disable_air_flow";
}
