$(document).ready(function() {
    $('#experiment_option').change(function() {
        var selectedOption = $(this).val();
        if (selectedOption) {
            $.ajax({
                url: '/get_functions',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({selected_option: selectedOption}),
                success: function(response) {
                    var functionDropdown = $('#function_option');
                    functionDropdown.empty();
                    $.each(response.functions, function(index, functionName) {
                        functionDropdown.append($('<option>', {
                            value: functionName,
                            text: functionName
                        }));
                    });
                },
                error: function(xhr, status, error) {
                    console.error("An error occurred: " + xhr.responseText);
                }
            });
        } else {
            $('#function_option').empty().append('<option value="">Select a file first</option>');
        }
    });

    $('#startExperiment').click(function() {
        var selectedOption = $('#experiment_option').val();
        var selectedFunction = $('#function_option').val();
        $.ajax({
            url: '/start_experiment',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({experiment_option: selectedOption, function_name: selectedFunction}),
            success: function(response) {
                $('#message').text(response.message);
            },
            error: function(xhr, status, error) {
                $('#message').text("An error occurred: " + xhr.responseText);
            }
        });
    });

    $('#stopExperiment').click(function() {
        $.ajax({
            url: '/stop_experiment',
            type: 'POST',
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
                    console.log("Updating chamber:", chamber.id);
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

                    var flyLocBox = $('#chamber-' + chamber.id + ' .current-fly-loc');
                    flyLocBox.text(chamber.currentFlyLoc);
                });

                response.odor_columns.forEach(function(odor_column) {
                    console.log("Updating odor column:", odor_column.id);
                    var airFlowBox = $('#odor-' + odor_column.id + ' .air-flow-box');
                    if (odor_column.airFlow) {
                        airFlowBox.removeClass('red').addClass('green');
                    } else {
                        airFlowBox.removeClass('green').addClass('red');
                    }

                    var leftOdorBox = $('#odor-' + odor_column.id + ' .left-odor-box');
                    if (odor_column.leftOdor) {
                        leftOdorBox.removeClass('red').addClass('green');
                    } else {
                        leftOdorBox.removeClass('green').addClass('red');
                    }

                    var rightOdorBox = $('#odor-' + odor_column.id + ' .right-odor-box');
                    if (odor_column.rightOdor) {
                        rightOdorBox.removeClass('red').addClass('green');
                    } else {
                        rightOdorBox.removeClass('green').addClass('red');
                    }
                });
            },
            error: function(xhr, status, error) {
                console.error("An error occurred while fetching chamber states: " + xhr.responseText);
            }
        });
    }

    function updateFlyPositions() {
        fetch('/get_fly_positions').then(response => response.json()).then(flyPositions => {
            document.querySelectorAll('.fly-marker').forEach(marker => marker.remove());
            flyPositions.forEach(position => {
                const marker = document.createElement('div');
                marker.className = flyMarkerClass;
                marker.style.left = (position[0] - 5) + 'px';
                marker.style.top = (position[1] - 5) + 'px';
                document.getElementById('video-feed').appendChild(marker);
            });
        });

        fetch('/get_fly_positions_vector').then(response => response.json()).then(data => {
            data.forEach((position, idx) => {
                const positionDiv = document.getElementById('fly-position-' + idx);
                positionDiv.textContent = position;
            });
        });

        updateChamberStates();
    }

    setInterval(updateFlyPositions, 1000); // Poll every 1 second
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

function showManualControl() {
    $('.popup-overlay').show();
    $('.popup').show();
}

function hideManualControl() {
    $('.popup-overlay').hide();
    $('.popup').hide();
}

let currentRect = null;
let currentResizeHandle = null;
let offsetX, offsetY, initialWidth, initialHeight;
let flyMarkerClass = 'fly-marker';

function handleVideoError() {
    console.error("Video feed failed to load.");
    alert("Video feed failed to load.");
}

function toggleFlyDetection() {
    const flyMarkers = document.querySelectorAll('.fly-marker');
    flyMarkerClass = flyMarkerClass === 'fly-marker' ? 'fly-marker-hidden' : 'fly-marker';
    flyMarkers.forEach(marker => {
        marker.className = flyMarkerClass;
    });
}

function toggleRectangles() {
    const rectangles = document.querySelectorAll('.rectangle');
    rectangles.forEach(rect => {
        if (rect.classList.contains('hidden-rectangle')) {
            rect.classList.remove('hidden-rectangle');
        } else {
            rect.classList.add('hidden-rectangle');
        }
    });
}

function startDrag(event, idx) {
    if (event.target.className.includes('resize-handle')) return;
    currentRect = document.getElementById('rect-' + idx);
    offsetX = event.clientX - currentRect.offsetLeft;
    offsetY = event.clientY - currentRect.offsetTop;
    document.addEventListener('mousemove', dragRect);
    document.addEventListener('mouseup', stopDrag);
}

function dragRect(event) {
    if (currentRect) {
        currentRect.style.left = (event.clientX - offsetX) + 'px';
        currentRect.style.top = (event.clientY - offsetY) + 'px';
    }
}

function stopDrag() {
    if (currentRect) {
        saveRectanglePosition();
        document.removeEventListener('mousemove', dragRect);
        document.removeEventListener('mouseup', stopDrag);
        currentRect = null;
    }
}

function startResize(event, idx) {
    currentResizeHandle = event.target;
    currentRect = document.getElementById('rect-' + idx);
    offsetX = event.clientX;
    offsetY = event.clientY;
    initialWidth = currentRect.offsetWidth;
    initialHeight = currentRect.offsetHeight;
    document.addEventListener('mousemove', resizeRect);
    document.addEventListener('mouseup', stopResize);
    event.stopPropagation();
}

function resizeRect(event) {
    if (currentRect) {
        const newWidth = Math.max(initialWidth + (event.clientX - offsetX), 10);
        const newHeight = Math.max(initialHeight + (event.clientY - offsetY), 10);
        currentRect.style.width = newWidth + 'px';
        currentRect.style.height = newHeight + 'px';
    }
}

function stopResize() {
    if (currentRect) {
        saveRectanglePosition();
        document.removeEventListener('mousemove', resizeRect);
        document.removeEventListener('mouseup', stopResize);
        currentRect = null;
        currentResizeHandle = null;
    }
}

function saveRectanglePosition() {
    if (currentRect) {
        let rectId = currentRect.id.split('-')[1];
        let rectData = {
            index: parseInt(rectId),
            x: currentRect.offsetLeft,
            y: currentRect.offsetTop,
            width: currentRect.offsetWidth,
            height: currentRect.offsetHeight
        };
        fetch('/update_rectangle', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(rectData)
        }).then(response => response.json()).then(data => {
            console.log(data);
        });
    }
}
