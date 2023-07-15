import React, { useEffect, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import YouTube from 'react-youtube'
import movieTrailer from 'movie-trailer'
import './MovieInfo.css'
import axios from 'axios'
import ReactStars from 'react-stars'

const base_url = 'https://image.tmdb.org/t/p/original/'

function MovieInfo() {
  const searchUrl = 'http://localhost:5000/recommend/'
  const [recommended,setRecommended] = useState([])
  let data=[]
  const [trailerUrl,setTrailerUrl] = useState('')
  const location = useLocation()
  // console.log(location.state.fromSerach)
  if(location.state.fromSearch){
    data=location.state.movie
  }else{
    data=location.state
  }
  console.log("hbehdscb",data)
  
  let opts = {
    height:'390px',
    width:'100%',
    playerVars:{
      autoplay:1
    }
  }

  let handleClick = (movie) =>{
    // console.log(movie)
    if(trailerUrl){
      setTrailerUrl('')
    }else{
      movieTrailer(movie?.name || movie?.title)
      .then((url) => {
        console.log(url)
        const urlParams = new URLSearchParams(new URL(url).search)
        setTrailerUrl(urlParams.get('v'))
      }).catch((e) =>alert('Unable to find trailer Try again later...'))
    }
  }
  let recommended_movie_list = []
  let recommended_movie = async(list)=>{
    recommended_movie_list = []
    for (let i = 0;i<list.length;i++){
      const response = await axios.get(`https://api.themoviedb.org/3/movie/${list[i]}?api_key=04c35731a5ee918f014970082a0088b1`);
      recommended_movie_list.push(response.data)
    }
    console.log("yufh",recommended_movie_list)
    setRecommended(recommended_movie_list)
  }
  useEffect(() => {
    async function fetchData() {
        const request = await axios.get(`/recommend/${data?.name || data?.title}`)
        recommended_movie(request.data)
    return request.data;
    }

  const list = fetchData();
  }, [data])

  return (
  <div style={{height:'100%'}}>
    <Link to='/'>
    {/* <img  
        className='movie__logo'
        src='https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/Netflix_2015_logo.svg/1920px-Netflix_2015_logo.svg.png'
        alt='Netflix Logo'
      /> */}
      <h1 style={{color:'red'}}>MovieRecap</h1>
    </Link>

    <div className='movieInfo'>
      <div style={{display:"flex" , paddingBottom:'30px'}}>
      <div className='movieInfo__banner'>
          <img 
              className='movieInfo__poster'
              src={`${base_url}${data.poster_path}`} 
              alt='Movie Banner' />
      </div>
      <div className='movieInfo__content'>
          <h1 className='movieInfo__title'>{data?.title || data?.name || data?.originam_name}</h1>
          <h2>{data.first_air_date}</h2>
          <h1>Overview</h1>
          <h1 className='movieInfo__description'>{data?.overview}</h1>
          <div><ReactStars
            value={data.vote_average/2}
            half={false}
            edit={false}
            count={5}
            size={24}
            color2={'#ffd700'} /></div>
          <div className='buttons'>
            <button className='btns' onClick={ ()  => handleClick(data)}>
                <i class="fa-solid fa-circle-play" />
                Play Trailer
            </button>
          </div>
      </div>
      </div>
      <div>
      {recommended.length>0?<div>
        <p style={{fontSize:'20px',fontWeight:'bold'}}>Recommended List</p>
        {recommended.map((movie, index)=>
                <Link className='row__link' to='/movieinfo' state={movie} key={index}>
                    <img className={`row__poster ${true && "row__posterLarge"}`}
                    key={movie.id}
                    src={`${base_url}${true? movie.poster_path:movie.backdrop_path}`} alt={movie.name}/>
                </Link>
            )}
      </div>:<p style={{fontSize:'20px',fontWeight:'bold'}}>No Recommended List</p>}
      </div>
      
    </div>
    {trailerUrl && <div className='movie__trailer'>
      <div className='div__btn'>
        <h2>{`${data.name?data.name:"Trailer"}`}</h2>
        <button className='close__btn' onClick={ ()  => handleClick(data)}>close</button>
      </div>
      <YouTube videoId={trailerUrl} opts={opts} />
    </div>}
  </div>
  )
}

export default MovieInfo