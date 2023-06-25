import axios from "axios";
import { useRef, useState } from "react";
import './App.css';
function App() {
  const MOTORBIKE = 0;
  const CAR = 1;
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);
  const [plates, setPlates] = useState(null)
  const [plateType, setPlateType] = useState(MOTORBIKE);
  const [showLoading, setShowLoading] = useState(false)
  const [showNotRecognize, setShowNotRecognize] = useState(false)
  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);

    // Preview the image
    const reader = new FileReader();
    reader.onload = () => {
      setPreviewImage(reader.result);
      setPlates([]);
      setShowNotRecognize(false);
    };
    reader.readAsDataURL(file);

    // Reset the file input value
    fileInputRef.current.value = null;
  };

  const handleUpload = () => {
    setShowLoading(true)
    if (selectedFile) {
      const formData = new FormData();
      formData.append('file', selectedFile);

      axios.post('/api/image', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        params: {
          plate_type: plateType, // Add the plate type flag
        },
      })
        .then(response => {
          setPlates(response.data?.plates);
          if (!response.data?.plates?.length) {
            setShowNotRecognize(true)
          }
          setShowLoading(false)
        })
        .catch(error => {
          console.error(error);
          setShowLoading(false)
        });

      // Reset the selected file after upload
      setSelectedFile(null);
      return;
    }
    if (!plateType) {
      alert("Please select type of plate")
      return setShowLoading(false)
    }
    setShowLoading(false)
    alert("Not file selected!")
  };

  const handlePlateTypeChange = (event) => {
    setPlateType(parseInt(event.target.value));
  };
  return (
    <div className="App">
      <div className="license-plate-recognition">
        <h1>License Plate Recognition</h1>
        <div className="plate-container">
          <div className="plate-type-container">
            <label>
              <input
                type="radio"
                name="plateType"
                value={MOTORBIKE}
                onChange={handlePlateTypeChange}
                checked={plateType === MOTORBIKE}
              />
              Motorbike Plate
            </label>
            <label>
              <input
                type="radio"
                name="plateType"
                value={CAR}
                onChange={handlePlateTypeChange}
                checked={plateType === CAR}
              />
              Car Plate
            </label>
          </div>
          {previewImage ? <img src={previewImage} alt="Preview" className="preview-plate" /> : <div className="placeholder">
            <p>No picture selected. Please select one picture to recognize!</p>
          </div>}
          {showLoading && <div className="lds-ring"><div></div><div></div><div></div><div></div></div>}
          {plates && plates.length > 0 && plates.map((plate, idx) => (
            plate && <div className="plate-number-group" key={idx}>
              <h2>Biển số xe:</h2>
              <p>{plate.replace(/[\[\]\(\)\|{}]/g, "")}</p>
            </div>
          ))}
          {showNotRecognize && <div className="placeholder">
            <p><strong>Can not recognize any plate in picture!</strong></p>
          </div>}
          <br />
        </div>
        <div className="button-group">
          <input type="file" ref={fileInputRef} hidden onChange={handleFileChange} />
          <button className="choose-file-btn" onClick={() => { fileInputRef.current.click() }}>Choose file</button>
          <button onClick={handleUpload}>Recognize Plate</button>
        </div>
      </div>
    </div >
  );
}

export default App;
