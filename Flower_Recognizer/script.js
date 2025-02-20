// Listen for file selection
document.getElementById('photoInput').addEventListener('change', function(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function(e) {
    // Show preview of the uploaded image
    document.getElementById('photoPreview').src = e.target.result;
    // Send file to server for processing
    sendImageToServer(file);
  };
  reader.readAsDataURL(file);
});

function sendImageToServer(file) {
  let formData = new FormData();
  formData.append('image', file);

  fetch('http://127.0.0.1:5000/upload', {
    method: 'POST',
    body: formData
  })
  .then(response => {
    console.log("Response status:", response.status);
    return response.text();
  })
  .then(data => {
    console.log('Server response:', data);
    document.getElementById('responseMessage').innerText = data;
  })
  .catch(error => {
    console.error('Error:', error);
    document.getElementById('responseMessage').innerText = "Error uploading file!";
  });
}
