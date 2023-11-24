import React, { useEffect } from 'react'
import { useState } from 'react'
import defaultImage from '../defaultImage.jpg'

export default function Form() {

    const [selectedImage, setSelectedImage] = useState(null)
    const [selectedImageURL, setSelectedImageURL] = useState(null)

    const fileValidate = (file) => {
        if (file.type === "image/jpg" || file.type === "image/jpeg") {
            return true;
        }
        else return false;
    }

    const handleChange = (e) => {
        console.log(e.target.files[0])

        if (e.target.files[0] === undefined) {
            setSelectedImage(null)
            setSelectedImageURL(null)

            return
        }

        if (fileValidate(e.target.files[0])) {
            setSelectedImage(e.target.files[0])
            setSelectedImageURL(URL.createObjectURL(e.target.files[0]))
        }
        else {
            alert('Please select the JPEG/JPG format of the image.')
        }
    }

    const formSubmitHandler = async (e) => {
        e.preventDefault();

        if (selectedImage === null) {
            alert('Please select an image before submitting.')
            return
        }

        const eModel = document.getElementById('model')
        const eNumberOfImages = document.getElementById('numberOfImages')

        if (eModel.value === 'select' || eNumberOfImages.value === 'select') {
            alert('Please select all the required fields')
            return
        }

        var request = new FormData();

        request.append("file", selectedImage);

        const response = await fetch("http://127.0.0.1:5001/search", {
            method: "POST",
            mode: "no-cors",
            body: request
        })

        console.log(response)

    }

    return (
        <div>
            <form onSubmit={formSubmitHandler} id='imageForm' encType='multipart/form-data'>
                <div className="mb-3 mx-4">
                    <label htmlFor="formFile" className="form-label">Please select a X-ray image</label>
                    <input className="form-control" type="file" id="formFile" name='file' onChange={handleChange} />
                </div>

                <div className='d-flex flex-row mb-4'>

                    <select id='model' name='model' className="form-select mx-4" aria-label="Select model">
                        <option value='select' selected>Select model</option>
                        <option value="lbp">Local Binary Pattern (LBP)</option>
                        <option value="vgg16">VGG-16</option>
                    </select>

                    <select id='numberOfImages' name='numberOfImages' className="form-select mx-4" aria-label="Number of similar images">
                        <option value='select' selected>Number of similar images</option>
                        <option value="10">10</option>
                        <option value="20">20</option>
                    </select>

                    <button type="submit" className="btn btn-success px-5 mx-4">Submit</button>
                </div>
            </form>

            <div>
                <div className='d-flex justify-content-center fs-4 mb-2'>Selected X-Ray Image</div>
                <div className='d-flex justify-content-center fs-4'>
                    <img src={selectedImageURL === null ? defaultImage : selectedImageURL} width={200} height={200} />
                </div>
            </div>
        </div>
    )
}
