document.addEventListener('DOMContentLoaded', function() {
    // Elements to update
    const ageElement = document.getElementById('age');
    const genderElement = document.getElementById('gender');
    const emotionElement = document.getElementById('emotion');
    const healthElement = document.getElementById('health');
    const distanceElement = document.getElementById('distance');
    const fpsElement = document.getElementById('fps');
    const objectsList = document.getElementById('objects-list');

    // Function to update the information
    function updateInfo() {
        fetch('/get_analysis')
            .then(response => response.json())
            .then(data => {
                ageElement.textContent = data.age || 'Unknown';
                genderElement.textContent = data.gender || 'Unknown';
                emotionElement.textContent = data.emotion || 'Unknown';
                healthElement.textContent = data.health || 'Unknown';
                distanceElement.textContent = data.distance || 'N/A';
                fpsElement.textContent = data.fps || 'N/A';
                
                if (data.objects && data.objects.length > 0) {
                    objectsList.textContent = data.objects.join(', ');
                } else {
                    objectsList.textContent = 'None detected';
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });
    }

    // Update every 500ms
    setInterval(updateInfo, 500);

    // Add some visual feedback when data updates
    function pulseElement(element) {
        element.classList.add('pulse');
        setTimeout(() => {
            element.classList.remove('pulse');
        }, 500);
    }

    // Error handling
    function handleError(error) {
        console.error('Error:', error);
        // Add visual feedback for errors
    }
}); 